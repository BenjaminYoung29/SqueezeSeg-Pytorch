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



