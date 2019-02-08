""" Bilateral Filter """

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BilateralFilter( nn.Module ):
    """ Computing pairwise energy with a bilateral filter for CRF

    Args:
        sizes: filter size for zenith and azimuth dimension
        stride: kernel strides
        padding: padding
    """

    def __init__( self, mc, stride=1, padding=0):
        super(BilateralFilter, self).__init__()

        self.mc = mc
        self.stride = stride
        self.padding = padding

    def forward( self, x ):
        mc = self.mc
        
        batch, in_channel, zenith, azimuth = list(x.size())
        size_z, size_a = mc.LCN_HEIGHT, mc.LCN_WIDTH

        condensing_kernel = torch.from_numpy(
                util.condensing_matrix(in_channel, size_z, size_a)
        ).float()

        condensed_input = F.conv2d(x, weight=condensing_kernel.to(device), stride=self.stride, padding=self.padding)

        diff_x = x[:, 0, :, :].view(batch, 1, zenith, azimuth) \
                - condensed_input[:, 0::in_channel, :, :]

        diff_y = x[:, 1, :, :].view(batch, 1, zenith, azimuth) \
                - condensed_input[ :, 1::in_channel, :, :]

        diff_z = x[:, 2, :, :].view(batch, 1, zenith, azimuth) \
                - condensed_input[ :, 2::in_channel, :, :]

        bi_filters = []

        for cls in range(mc.NUM_CLASS):
            theta_r = mc.BILATERAL_THETA_R[cls]
            bi_filter = torch.exp( - (diff_x**2 + diff_y**2 + diff_z**2) / 2 / theta_r**2 )
            bi_filters.append(bi_filter)

        bf_weight = torch.stack(bi_filters)
        bf_weight = bf_weight.transpose(0,1)

        return bf_weight



