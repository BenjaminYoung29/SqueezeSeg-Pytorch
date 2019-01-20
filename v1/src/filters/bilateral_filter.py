""" Bilateral Filter """

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util

class BilateralFilter( nn.Module ):
    """ Computing pairwise energy with a bilateral filter for CRF

    Args:
        thetas: theta parameter for bilateral filter
        sizes: filter size for zenith and azimuth dimension
        stride: kernel strides
        padding: padding
    """

    def __init__( self, thetas=[0.9, 0.01], sizes=[3, 5], stride=1, padding=0):
        self.theta_a, self.theta_r = thetas
        self.size_z, self.size_a = sizes
        self.pad_z, self.pad_a = self.size_z//2, self.size_a//2

    def forward( self, x ):
        batch, zenith, azimuth, in_channel = x.shape.as_list()

        condensing_kernel = util.condensing_matrix(self.size_z, self.size_a, in_channel)

        condensed_input = 


