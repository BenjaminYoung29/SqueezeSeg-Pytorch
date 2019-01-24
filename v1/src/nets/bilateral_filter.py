""" Bilateral Filter """

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util

class BilateralFilter( nn.Module ):
    """ Computing pairwise energy with a bilateral filter for CRF

    Args:
        sizes: filter size for zenith and azimuth dimension
        stride: kernel strides
        padding: padding
    """

    def __init__( self, mc, stride=1, padding=0):
        
        self.mc = mc
        self.stride = stride
        self.padding = padding

    def forward( self, x ):
        mc = self.mc
        
        batch, zenith, azimuth, in_channel = list(x.size())
        size_z, size_a = mc.LCN_HEIGHT, mc.LCN_WIDTH

        condensing_kernel = util.condensing_matrix(size_z, size_a, in_channel)

        condensed_input = F.conv2d(x, filter=condensing_kernel, stride=self.stride, padding=self.padding)

        diff_x = x[:, :, :, 0].view(batch, zenith, azimuth, 1) \
                - condensed_input[:, :, :, 0::in_channel]

        diff_y = x[:, :, :, 1].view(batch, zenith, azimuth, 1) \
                - condensed_input[ :, :, :, 1::in_channel]

        diff_z = x[:, :, :, 2].view(batch, zenith, azimuth, 1) \
                - condensed_input[ :, :, :, 2::in_channel]

        bi_filters = []

        # NUM_CLASSやBILATERAL_THETA_A,BILATERAL_THETA_Rはあとでconfigの設定とかでなんとかする
        # configparserを用いるといい
        for cls in range(mc.NUM_CLASS):
            theta_r = mc.BILATERAL_THETA_R[cls]
            bi_filter = torch.exp( - (diff_x**2 + diff_y**2 + diff_z**2) / 2 / theta_r**2 )
            bi_filters.append(bi_filter)

        out = torch.stack(bi_filters)
        out = out.transpose(0,1).transpose(1,2).transpose(2,3).transpose(3,4)

        return out



