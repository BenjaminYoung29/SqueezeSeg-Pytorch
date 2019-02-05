""" Train """

import argparse
import os.path
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from config import *
from datasets import * 
from utils import *
from nets import *

parser = argparse.ArgumentParser(description='Pytorch SqueezeSeg Training')

parser.add_argument('--csv_path', default='../../data/ImageSet/csv/', type=str, help='path to where csv file')
parser.add_argument('--dir_path', default='../../data/lidar_2d/', type=str, help='path to where data')

# Hyper Params
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optim')
parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')

# Device Option
parser.add_argument('--gpu_ids', default=[0,1], type=int, nargs="+", help='which gpu you use')
parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class SqueezeSegLoss( nn.Module ):
    def __init__(self, mc):
        super(SqueezeSegLoss, self).__init__()
        self.mc = mc
        
    def forward(self, outputs, targets, lidar_mask, loss_weight):
        mc = self.mc

        loss = F.cross_entropy(outputs.view(-1, mc.NUM_CLASS), targets.view(-1,))
        loss = lidar_mask.view(-1,) * loss
        loss = loss_weight.view(-1,) * loss
        loss = torch.sum(loss) / torch.sum(lidar_mask) * mc.CLS_LOSS_COEF

        return loss 


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()

    running_loss = 0.0
    
    for i, datas in enumerate(train_loader):
        inputs, mask, targets, weight = datas
        inputs, mask, targets, weight = \
                inputs.to(device), mask.to(device), targets.to(device), weight.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, mask)
        loss = criterion(outputs, targets, mask, weight)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f'[{epoch+1}, {i+1:05}] loss: {running_loss/10:.3f}')
            running_loss = 0.0

def validate(model, val_loader, epoch):
    model.eval()
    
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, datas in enumerate(val_loader):
            inputs, mask, targets, weight = datas
            inputs, mask, targets, weight = \
                    inputs.to(device), mask.to(device), targets.to(device), weight.to(device)

            outputs = model(inputs, mask)
            print(f'Outputs: {outputs}')
            
            _, predicted = torch.max(outputs.data, 1)
            
            #total += labels.size(0)
            #correct += (predicted == labels).sum()

    #print(f'Accuracy of the network on the test images: {100 * correct / total}')

if __name__ == '__main__':
    mc = kitti_squeezeSeg_config()
    
    # train data 読み込み
    train_datasets = KittiDataset(
        mc, 
        csv_file = args.csv_path + 'train.csv', 
        root_dir = args.dir_path, 
        transform = transforms.Compose([transforms.ToTensor()])
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # val data 読み込み
    val_datasets = KittiDataset(
        mc,
        csv_file = args.csv_path + 'val.csv',
        root_dir = args.dir_path,
        transform = transforms.Compose([transforms.ToTensor()])
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    model = SqueezeSeg(mc).to(device)
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    
    # Lossは自作する
    criterion = SqueezeSegLoss(mc)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(args.epochs):
        scheduler.step()
        train(model, train_dataloader, criterion, optimizer, epoch)
        validate(model, val_dataloader, epoch) 
