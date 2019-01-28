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
# from imdb import kitti
from datasets.dataset import *
from utils.util import *
from nets import squeezeSeg

parser = argparse.ArgumentParser(description='Pytorch SqueezeSeg Training')

parser.add_argument('--csv_path', default='/data/csv/', type=str, help='path to where csv file')
parser.add_argument('--dir_path', default='/data/lidar_2d/', type=str, help='path to where data')

# Hyper Params
parser.add_argument('--lr', default=1., type=float, help='initial learning rate')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')

# Device Option
parser.add_argument('--gpu_ids', default=[0,1], type=int, nargs="+", help='which gpu you use')
parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 



def train(model, train_loader, criterion, optimizer, epoch):
    model.train()

    for batch_idx, datas in enumerate(train_loader):
        inputs, mask, targets, weight = datas
        inputs, mask, targets, weight = \
                inputs.to(device), mask.to(device), targets.to(device), weight.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def validate(model, val_loader, epoch):
    model.eval()

    for batch_idx, datas in enumerate(val_loader):
        inputs, mask, targets, weight = datas
        inputs, mask, targets, weight = \
                inputs.to(device), mask.to(device), targets.to(device), weight.to(device)

        outputs = model(inputs)
        loss = crterion(outputs, targets)

        print(loss)

if __name__ == '__main__':
    mc = kitti_squeezeSeg_config()
    
    # train data 読み込み
    train_datasets = KittiDataset(
        mc, 
        csv_file = args.csv_path + 'train.csv', 
        root_dir = args.dir_ath, 
        transform = transforms.Compose([ToTensor()])
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # val data 読み込み
    val_datasets = KittiDataset(
        mc,
        csv_file = args.csv_path + 'val.csv',
        root_dir = args.dir_path,
        transform = transforms.Compose([ToTensor()])
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = SueezeSeg(mc).to(device)
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lerning_rate)

    for epoch in range(args.epochs):
        train(model, train_dataloader, criterion, optimier, epoch)
    
