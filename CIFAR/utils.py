'''
    setup model and datasets
'''




import copy 
import torch
import numpy as np 
from advertorch.utils import NormalizeByChannelMeanStd

from models import *
from dataset import *


__all__ = ['setup_model_dataset',"setup_model"]

def setup_model(args):
    
    if args.dataset == 'cifar10':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    elif args.dataset == 'cifar100':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])

    else:
        raise ValueError('Dataset not supprot yet !')

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    # print(model)

    return model


def setup_model_dataset(args):

    if args.dataset == 'cifar10':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)

    elif args.dataset == 'cifar100':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
    
    else:
        raise ValueError('Dataset not supprot yet !')

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    # print(model)

    return model, train_set_loader, val_loader, test_loader


