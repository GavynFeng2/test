#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse
from collections import OrderedDict
from matplotlib import pyplot as plt

import torch

from conf import settings
from utils import get_network, get_dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu device id, set `-1` to use cpu only')
    parser.add_argument('--batch', '-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    if args.gpu == '-1':
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            raise ValueError('GPU is not available. please set `--gpu -1` to use cpu only. ')
        
    net = get_network(args.net).to(device)

    _, cifar100_test_loader = get_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        rank=0,
        num_workers=4,
        batch_size=args.batch,
    )

    state_dict = torch.load(args.weights, weights_only=True)
    new_state_dict = OrderedDict()

    # If training and saving model with DDP, it's necessary to remove the prefix `module.`
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            image, label = image.to(device), label.to(device)

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if device != 'cpu':
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
