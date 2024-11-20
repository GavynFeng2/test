from cangjie_utils_b3 import get_training_loader,get_val_loader
from models.squeezenet import squeezenet
from time import perf_counter
import argparse
import os
# from utils import WarmUpLR

import torch
import torch.nn as nn
from conf import settings

from utils import  WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from cangjie_models.sqnetR4 import sqnetr4
from cangjie_models.sqnetR4C3579 import sqnetr4c3579
from cangjie_models.sqnetD4 import sqnetd4
from cangjie_models.sqnetD4C3579 import sqnetd4c3579
from cangjie_models.sqnetR4C3579RNN import sqnetr4c3579rnn

def train(epoch):
    start_time = perf_counter()
    net.train()
    for batch_index, (images, labels, string_labels_padded, target_lengths) in enumerate(ETL952TrainLoader):

        if args.gpu:
            labels = labels.cuda().long()
            images = images.cuda()
            string_labels_padded = string_labels_padded.cuda()
            target_lengths = target_lengths.cuda()
        else:
            labels = labels.long()
            target_lengths = target_lengths.long()



        optimizer.zero_grad()
        x_cls, x_seq = net(images)
        # loss = loss_function(outputs, labels)
        # loss.backward()
        # optimizer.step()

        

        loss_class = criterion_class(x_cls, labels)

        batch_size = x_seq.size(0)
        sequence_length = x_seq.size(1)
        input_lengths = torch.full((batch_size,), sequence_length, dtype=torch.long)
        print("x_seq shape:", x_seq.shape)
        print("string_labels_padded shape:", string_labels_padded.shape)
        print("input_lengths shape:", input_lengths.shape)
        print("target_lengths shape:", target_lengths.shape)
        loss_seq = criterion_seq(x_seq.log_softmax(2), string_labels_padded, input_lengths, target_lengths)
        
        loss = loss_class + loss_seq
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(ETL952TrainLoader) + batch_index + 1    
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        print('Training Epoch: {epoch}/{total_epochs} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            total_epochs=settings.EPOCH,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(ETL952TrainLoader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish_time = perf_counter()
    with open('train.log', 'a') as f:
        f.write('epoch {} training time consumed: {:.2f}s\n'.format(epoch, finish_time - start_time))
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish_time - start_time))

import torch
from time import perf_counter
from torch.nn.utils.rnn import pad_sequence

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start_time = perf_counter()
    net.eval()
    test_loss_class = 0.0  # Classification loss
    test_loss_seq = 0.0    # Sequence loss
    correct = 0.0
    total_sequences = 0
    correct_sequences = 0

    for batch_index, (images, class_labels, string_labels_padded, target_lengths) in enumerate(ETL952ValLoader):
        if torch.cuda.is_available():
            images = images.cuda()
            class_labels = class_labels.cuda().long()
            string_labels_padded = string_labels_padded.cuda()
            target_lengths = target_lengths.cuda()
        
        # Forward pass
        outputs_class, outputs_seq = net(images)
        
        # Classification loss
        loss_class = criterion_class(outputs_class, class_labels)
        
        # Sequence loss
        loss_seq = criterion_seq(outputs_seq.log_softmax(2), string_labels_padded, torch.full((outputs_seq.size(0),), outputs_seq.size(1), dtype=torch.long).to("cuda"), target_lengths)
        
        # Accumulate losses
        test_loss_class += loss_class.item()
        test_loss_seq += loss_seq.item()
        
        # Calculate accuracy
        _, preds = outputs_class.max(1)
        correct += preds.eq(class_labels).sum().item()
        
        # Evaluate sequence prediction accuracy (pseudo code, needs actual implementation)
        # Here we would compare the predicted sequences with the ground truth sequences
        # This part of accuracy calculation is simplified and needs to be replaced with actual evaluation logic
        for i in range(len(string_labels_padded)):
            predicted_sequence = outputs_seq.argmax(dim=2)[i]
            target_sequence = string_labels_padded[i, :target_lengths[i]]
            if torch.equal(predicted_sequence[:len(target_sequence)], target_sequence):
                correct_sequences += 1
            total_sequences += 1

    finish_time = perf_counter()
    
    # Print results
    print('Evaluating Network.....')
    print('Epoch: {}, Classification Test Loss: {:.4f}, Sequence Test Loss: {:.4f}, Test Accuracy: {:.4f}, Sequence Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
        epoch, 
        test_loss_class / len(ETL952ValLoader.dataset),
        test_loss_seq / len(ETL952ValLoader.dataset),
        correct / len(ETL952ValLoader.dataset),
        correct_sequences / total_sequences if total_sequences > 0 else 0,
        finish_time - start_time
    ))
    
    # TensorBoard logging
    if tb:
        writer.add_scalar('Test/Classification Average loss', test_loss_class / len(ETL952ValLoader.dataset), epoch)
        writer.add_scalar('Test/Sequence Average loss', test_loss_seq / len(ETL952ValLoader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct / len(ETL952ValLoader.dataset), epoch)
        writer.add_scalar('Test/Sequence Accuracy', correct_sequences / total_sequences if total_sequences > 0 else 0, epoch)
    
    return correct / len(ETL952ValLoader.dataset), correct_sequences / total_sequences if total_sequences > 0 else 0


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', default='sqnetr4c3579rnn', type=str, help='net type')
    parser.add_argument('-gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    current_time = settings.TIME_NOW

    

    #model
    net = sqnetr4c3579rnn()
    file_name = "net_ver"+current_time+".log"
    net_ver_path = os.path.join('net_ver',file_name)
    if not os.path.exists('net_ver'):
        os.mkdir('net_ver')
    with open(net_ver_path, "w") as f:
        f.write(str(net))
    print("assigned model")

    #data
    time_in = perf_counter()
    print("start loading data")
    ETL952TrainLoader = get_training_loader(batch_size=args.batch_size)
    ETL952ValLoader = get_val_loader(batch_size=args.batch_size)
    print("data loaded")
    time_out = perf_counter()
    with open("train.log", "a") as f:
        f.write("net_ver"+ current_time + " ")
        f.write("train at" + current_time + " ")
        f.write("data loading time: {:.2f}s\n".format(time_out - time_in))


    #setup model

    # loss_function = torch.nn.CrossEntropyLoss()
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_seq = torch.nn.CTCLoss(blank=0)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(ETL952TrainLoader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    
    

    # def ctc_loss(y_true, y_pred, input_lengths, target_lengths):
    #     return F.ctc_loss(y_pred, y_true, input_lengths, target_lengths)

    # def combined_loss(class_pred, class_true, seq_pred, seq_true, input_lengths, target_lengths):
    #     classification_loss = F.cross_entropy(class_pred, class_true)
    #     sequence_loss = ctc_loss(seq_true, seq_pred, input_lengths, target_lengths)
    #     return classification_loss + sequence_loss

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
            settings.LOG_DIR, args.net, current_time))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
        net = net.cuda()
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

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    time_in = perf_counter()
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

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
    time_out = perf_counter()
    with open("train.log", "a") as f:
        f.write("train at" + current_time + " ")
        f.write("train time: {:.2f}s\n".format(time_out - time_in))
    writer.close()