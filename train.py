# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, \
    best_acc_weights, reduce_metric

def train(epoch):
    start = time.time()
    net.train()
    step_times = []
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        if batch_index > 0 and batch_index < len(cifar100_training_loader) - 1:
            step_time = time.time() - start
            step_times.append(step_time)
        
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        
        if master_process:
            print('Training Epoch: {epoch} [{train_step}/{total_step}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                train_step=(batch_index + 1),
                total_step=len(cifar100_training_loader)
            ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warmup:
            warmup_scheduler.step()

    if master_process:
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    
    avg_step_time = sum(step_times) / len(step_times) if len(step_times) > 0 else 0
    avg_throughput = len(cifar100_training_loader.dataset) / avg_step_time if avg_step_time > 0 else 0

    epoch_duration = time.time() - start
    if master_process:
        print('epoch {} training time consumed: {:.2f}s, avg_throughput: {:.2f} imgs/sec'.format(
        epoch, epoch_duration, avg_throughput))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    if ddp:
        correct = correct.clone().detach().to(device)
        correct = reduce_metric(correct)
        correct = 100. * correct.item()
            
    finish = time.time()

    if master_process:
        if device != 'cpu':
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')
        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(cifar100_test_loader),
            correct / len(cifar100_test_loader.dataset),
            finish - start
        ))
        print()

    #add informations to tensorboard
    if tb and master_process:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader), epoch)
        writer.add_scalar('Test/Accuracy', correct / len(cifar100_test_loader.dataset), epoch)

    return correct / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--gpu', type=str, default='-1', help='gpu device id, set `-1` to use cpu only')
    parser.add_argument('--batch', '-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--epoch', type=int, default=100, help='training epochs')
    parser.add_argument('--warmup', type=int, default=1, help='warm up training phase')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer name')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers for dataloader')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    if args.gpu == '-1':
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        else:
            raise ValueError('GPU is not available. please set `--gpu -1` to use cpu only. ')
    
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f'cuda:{local_rank}')
        master_process = rank == 0
        seed_offset = rank
    else:
        master_process = True
        rank = 0
        seed_offset = 0
        world_size = 1

    torch.manual_seed(2024 + seed_offset)

    net = get_network(args.net).to(device)
    if ddp:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    
    #data preprocessing:
    cifar100_training_loader, cifar100_test_loader = get_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        rank=rank,
        batch_size=args.batch,
        num_workers=args.num_workers,
        const_test_batch=True,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optimizer, 'SGD')(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log

    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32).to(device)
    if ddp:
        if dist.get_rank() == 0:
            writer.add_graph(net.module, input_tensor)
    else:
        writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))
        if ddp:
            net = DDP(net, device_ids=[local_rank], output_device=local_rank)
        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, args.epoch + 1):
        if epoch > args.warmup:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        if master_process:
            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)

        if ddp:
            dist.barrier()

    writer.close()
    if ddp:
        dist.destroy_process_group()

