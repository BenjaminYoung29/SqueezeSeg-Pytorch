""" Train """

import argparse
from datetime import absolute_import
import os.path
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd

from config import *
from imdb import kitti
from utils.util import *
from nets import squeezeSeg

parser = argparse.ArgumentParser(description='Pytorch SqueezeSeg Training')

# Hyper Params
parser.add_argument('--lr', default=1., type=float, help='initial learning rate')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')

# Device Option
parser.add_argument('--gpu_ids', default=[0,1], type=int, nargs="+", help='which gpu you use')
parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size')

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 学習データのロード周りもしっかりと実装みてから

def train(model, criterion, optimizer, epoch):
    model.train()

    for batch_idx, (datas, targets) in enumerate(train_loader):
        datas, targets = datas.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(datas)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    mc = kitti_squeezeSeg_config()
    model = SueezeSeg(mc).to(device)
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lerning_rate)

    for epoch in range(args.epochs):
        train(model, criterion, optimier, epoch)
    
