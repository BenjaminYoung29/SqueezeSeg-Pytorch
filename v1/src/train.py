""" Train """

import argparse
import os.path
import sys
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

from config import *
from datasets import *
from utils import *
from nets import *

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Pytorch SqueezeSeg Training')

parser.add_argument('--csv_path', default='../data/ImageSet/csv/', type=str, help='path to where csv file')
parser.add_argument('--data_path', default='../data/lidar_2d/', type=str, help='path to where data')
parser.add_argument('--model_path', default='./model', type=str, help='path to where model')

# Hyper Params
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optim')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay for optim')
parser.add_argument('--lr_step', default=1000, type=int, help='number of lr step')
parser.add_argument('--lr_gamma', default=0.1, type=float, help='gamma for lr scheduler')

parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='number of epoch to start learning')
parser.add_argument('--pretrain', default=False, type=bool, help='Whether or not to pretrain')
parser.add_argument('--resume', default=False, type=bool, help='Whether or not to resume')

# Device Option
parser.add_argument('--gpu_ids', dest='gpu_ids', default=[0,1], nargs="+", type=int, help='which gpu you use')
parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size')

args = parser.parse_args()

# To use TensorboardX
writer = SummaryWriter()

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
        loss = torch.sum(loss) / torch.sum(lidar_mask)

        return loss * mc.CLS_LOSS_COEF


def train(model, train_loader, criterion, optimizer, epoch):

    model.train()

    total_loss = 0.0
    total_size = 0.0

    for batch_idx, datas in enumerate(train_loader, 1):
        inputs, mask, targets, weight = datas
        inputs, mask, targets, weight = \
                inputs.to(device), mask.to(device), targets.to(device), weight.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, mask)

        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, targets, mask, weight)
        writer.add_scalar('data/loss', loss/args.batch_size, batch_idx * (epoch+1))

        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        total_size += inputs.size(0)
        if batch_idx % 100 == 0:
            now = datetime.datetime.now()

            print(f'[{now}] Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tAverage loss: {total_loss / total_size:.6f}')

            # TensoorboardX Save Input Image and Visualized Segmentation
            writer.add_image('Input/Image/', (img_normalize(inputs[0, 3, :, :])).cpu(), batch_idx * (epoch+1))

            writer.add_image('Predict/Image/', visualize_seg(predicted, mc)[0], batch_idx * (epoch+1))

            writer.add_image('Target/Image/', visualize_seg(targets, mc)[0], batch_idx * (epoch+1))


def test(mc, model, val_loader, epoch):

    model.eval()

    total_tp = np.zeros(mc.NUM_CLASS)
    total_fp = np.zeros(mc.NUM_CLASS)
    total_fn = np.zeros(mc.NUM_CLASS)

    with torch.no_grad():
        for batch_idx, datas in enumerate(val_loader):
            inputs, mask, targets, weight = datas
            inputs, mask, targets, weight = \
                    inputs.to(device), mask.to(device), targets.to(device), weight.to(device)

            outputs = model(inputs, mask)

            _, predicted = torch.max(outputs.data, 1)

            tp, fp, fn = evaluate(targets, predicted, mc.NUM_CLASS)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        iou = total_tp / (total_tp+total_fn+total_fp+1e-12)
        precision = total_tp / (total_tp+total_fp+1e-12)
        recall = total_tp / (total_tp+total_fn+1e-12)

        print()
        print_evaluate(mc, 'IoU', iou)
        print_evaluate(mc, 'Precision', precision)
        print_evaluate(mc, 'Recall', recall)
        print()


if __name__ == '__main__':
    mc = kitti_squeezeSeg_config()

    if os.path.exists(args.model_path) is False:
        os.mkdir(args.model_path)

    # train data 読み込み
    #print(f'augmentation: {mc.DATA_AUGMENTATION}')
    train_datasets = KittiDataset(
        mc,
        csv_file = args.csv_path + 'train.csv',
        root_dir = args.data_path,
        transform = transforms.Compose([transforms.ToTensor()])
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
        root_dir = args.data_path,
        transform = transforms.Compose([transforms.ToTensor()])
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = SqueezeSeg(mc).to(device)

    if args.pretrain:
        if args.resume:
            load_checkpoint(args.model_dir, args.start_epoch - 1, model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = SqueezeSegLoss(mc)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        print('-------------------------------------------------------------------')
        train(model, train_dataloader, criterion, optimizer, epoch)
        test(mc, model, val_dataloader, epoch)
        save_checkpoint(args.model_path, epoch, model)
        print('-------------------------------------------------------------------')
        print()

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
