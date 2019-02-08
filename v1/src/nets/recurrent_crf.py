""" Recurrent CRF """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RecurrentCRF( nn.Module ):
    def __init__( self, mc, stride=1, padding=0):
        super(RecurrentCRF, self).__init__()

        self.mc = mc
        self.stride = stride
        self.padding = padding

    def locally_connected_layer(self, inputs, lidar_mask, bilateral_filters, angular_filters, bi_angular_filters, condensing_kernel):
        mc = self.mc

        size_z, size_a = mc.LCN_HEIGHT, mc.LCN_WIDTH
        pad_z, pas_a = size_z//2, size_a//2
        half_filter_dim = (size_z*size_a)//2
        batch, in_channel, zenith, azimuth = list(inputs.size())
        
        ang_output = F.conv2d(inputs, weight=angular_filters, stride=self.stride, padding=self.padding)

        bi_ang_output = F.conv2d(inputs, weight=bi_angular_filters, stride=self.stride, padding=self.padding)

        condensed_input = F.conv2d(
                inputs * lidar_mask, weight=condensing_kernel, stride=self.stride, padding=self.padding
        )
        condensed_input = condensed_input.view(
                batch, in_channel, size_z * size_a -1, zenith, azimuth
        )
        condensed_input = torch.sum((condensed_input * bilateral_filters), 2)

        bi_output = torch.mul(condensed_input, lidar_mask)
        bi_output *= bi_ang_output
        
        return ang_output, bi_output


    def forward( self, x, lidar_mask, bilateral_filters ):

        mc = self.mc
        size_z, size_a = mc.LCN_HEIGHT, mc.LCN_WIDTH

        # initialize compatibilty matrices
        compat_kernel_init = torch.from_numpy(
                np.reshape(
                    np.ones((mc.NUM_CLASS, mc.NUM_CLASS), dtype="float32") - np.identity(mc.NUM_CLASS, dtype="float32"),
                    [mc.NUM_CLASS, mc.NUM_CLASS, 1, 1]
                )
        )

        bi_compat_kernel = compat_kernel_init * mc.BI_FILTER_COEF
        bi_compat_kernel.requires_grad_()

        angular_compat_kernel = compat_kernel_init * mc.ANG_FILTER_COEF
        angular_compat_kernel.requires_grad_()

        condensing_kernel = torch.from_numpy(
                util.condensing_matrix(mc.NUM_CLASS, size_z, size_a)
        ).float()

        angular_filters = torch.from_numpy(
                util.angular_filter_kernel(mc.NUM_CLASS, size_z, size_a, mc.ANG_THETA_A**2)
        ).float()

        bi_angular_filters = torch.from_numpy(
                util.angular_filter_kernel(mc.NUM_CLASS, size_z, size_a, mc.BILATERAL_THETA_A**2)
        ).float()

        # GPU
        bi_compat_kernel, angular_compat_kernel, condensing_kernel, angular_filters, bi_angular_filters = \
                bi_compat_kernel.to(device), angular_compat_kernel.to(device), condensing_kernel.to(device), angular_filters.to(device), bi_angular_filters.to(device)

        for it in range(mc.RCRF_ITER):
            unary = F.softmax(x, dim=-1)

            ang_output, bi_output = self.locally_connected_layer(
                    unary, lidar_mask, bilateral_filters, angular_filters, bi_angular_filters, condensing_kernel
            )

            ang_output = F.conv2d(ang_output, weight=angular_compat_kernel, stride=self.stride, padding=0)

            bi_output = F.conv2d(bi_output, weight=bi_compat_kernel, stride=self.stride, padding=0)

            pairwise = torch.add(ang_output, bi_output)

            outputs = torch.add(unary, pairwise)

            x = outputs
        
        return outputs
