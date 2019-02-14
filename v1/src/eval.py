""" Evaluation """

import argparse
import os
import sys

import torch
import torch.nn as nn
from torchvision import datasets,transforms
import torch.beckends.cudnn as cudnn

from config import *
from datasets import *
from utils import *
from nets import *

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Pytorch SqueezeSeg Evaluation')

parser.add_argument('--csv_path', default='../data/ImageSet/csv/', type=str, help='path to where csv file')
parser.add_argument('--data_path', default='../data/lidar_2d/', type=str, help='path to where data')
parser.add_argument('--model_path', default='./model', type=str, help='path to where model')
parser.add_argument('--model_num', default=0, type=int, help='number of model')

arser.add_argument('--gpu_ids', dest='gpu_ids', default=[0,1], nargs="+", type=int, help='which gpu you use')

args = parser.parse_args() 

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(mc, model, test_loader):
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

        print('--------------------------------------------------------------------------------')
        print()
        print_evaluate(mc, 'IoU', iou)
        print_evaluate(mc, 'Precision', precision)
        print_evaluate(mc, 'Recall', recall)
        print('--------------------------------------------------------------------------------')


if __name__ == '__main__':
    mc = kitti_squeezeSeg_config()
    mc.DATA_AUGMENTATION = False

    test_datasets = KittiDataset(
        mc,
        csv_file = args.csv_path + 'test.csv',
        root_dir = args.data_path,
        transform = transforms.Compose([transforms.ToTensor()])
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    model = SqueezeSeg(mc).to(device)

    load_checkpoint(args.model_dir, args.model, model)
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    test(mc, model, test_dataloader)


