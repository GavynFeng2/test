#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from cangjie_utils import  get_test_loader
from models.squeezenet import squeezenet
from cangjie_models.sqnetR import sqnetr
from cangjie_models.sqnetC9 import sqnetc9
from cangjie_models.sqnetC3579 import sqnetc3579
from cangjie_models.sqnetF4C3579 import sqnetf4c3579
from cangjie_models.sqnetR4 import sqnetr4
from cangjie_models.sqnetR4C3579 import sqnetr4c3579
from cangjie_models.sqnetD4 import sqnetd4
from cangjie_models.sqnetD4C3579 import sqnetd4c3579
from utils import get_network
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sqnetd4c3579', help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    args = parser.parse_args()

    net = sqnetd4c3579()
    if args.gpu:
        net = net.cuda()
    
    
    ETL952TestLoader = get_test_loader(
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
     
    cnt = np.ndarray([952, 952])

    with torch.no_grad():
        
        for n_iter, (image, label) in enumerate(ETL952TestLoader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(ETL952TestLoader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')


            output = net(image)
            # print(output.shape)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(ETL952TestLoader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(ETL952TestLoader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


# from models.squeezenet import SqueezeNet

# model = SqueezeNet(class_num=100)
# model.load_state_dict(torch.load('checkpoint\squeezenet\Saturday_13_July_2024_18h_05m_14s\squeezenet-200-regular.pth'))