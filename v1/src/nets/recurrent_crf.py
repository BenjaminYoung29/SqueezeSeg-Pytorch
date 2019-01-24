""" Recurrent CRF """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util

class RecurrentCRF( nn.Module ):
    def __init__( self, mc, stride=1, padding=0):
        self.mc = mc
        self.stride = stride
        self.padding = padding

        # model parameter
        self.model_params = []
        # ここもあとでparameterで変える
        self.lider_mask = torch.zeros([mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],dtype=torch.float32)
    
    def locally_connected_layer(self, inputs, bilateral_filters, angular_filters, bi_angular_filters, condensing_kernel):
        # "LIDAR MASK"のところはあとで考える

        mc = self.mc

        size_z, size_a = mc.LCN_HEIGHT, mc.LCN_WIDTH
        pad_z, pas_a = size_z//2, size_a//2
        half_filter_dim = (size_z*size_a)//2
        batch, zenith, azimuth, in_channel = list(inputs.size())
        
        ang_output = F.conv2d(inputs, filter=angular_filters, stride=self.stride, padding=self.padding)

        bi_ang_output = F.conv2d(inputs, filter=bi_angular_filters, stride=self.stride, padding=self.padding)

        condensed_input = F.conv2d(
                inputs*"LIDAR_MASK", filter=condensing_kernel, stride=self.stride, padding=self.padding
        )
        condensed_input = condensed_input.view(
                batch, zenith, azimuth, size_z*size_a-1, in_channel
        )
        condensed_input_np = condensed_input.numpy(condensed_input * bilateral_filters)
        condensed_input = torch.from_numpy(np.sum(condensed_input_np, axis=3))

        bi_output =   condensed_input * "LIDAR_MASK"
        bi_output *= bi_ang_output

        return ang_output, bi_output


    def forward( self, x, bilateral_filters ):
        mc = self.mc

        # initialize compatibilty matrices
        compat_kernel_init = torch.from_numpy(
                np.reshape(
                    np.ones((mc.NUM_CLASS, mc.NUM_CLASS), dtype="float32") - np.identity(mc.NUM_CLASS, dtype="float32"),
                    [1, 1, mc.NUM_CLASS, mc.NUM_CLASS]
                )
        )

        bi_compat_kernel = compat_kernel_init * mc.BI_FILTER_COEF
        bi_compat_kernel.requires_grad_()

        angular_compat_kernel = compat_kernel_init * mc.ANG_FILTER_COEF
        angular_compat_kernel.requires_grad_()

        self.model_params += [bi_compat_kernal, angular_compat_kernel]

        condensing_kernel = torch.from_numpy(
                util.condensing_matrix(self.sizes[0], self.sizes[1], mc.NUM_CLASS)
        )

        angular_filters = torch.from_numpy(
                util.angular_filter_kernel(self.sizes[0], self.sizes[1], mc.NUM_CLASS, mc.ANG_THETA_A**2)
        )

        bi_angular_filters = torch.from_numpy(
                util.angular_filter_kernel(self.sizes[0], self.sizes[1], mc.NUM_CLASS, mc.BILATERAL_THETA_A**2)
        )

        for it in range(mc.RCRF_ITER):
            unary = F.softmax(x, dim=-1)

            ang_output, bi_output = self.locally_connected_layer(
                    unary, bilateral_filters, angular_filters, bi_angular_filters, condensing_kernel
            )

            ang_output = F.conv2d(ang_output, filter=angular_compat_kernel, stride=self.stride, padding=self.padding)

            bi_output = F.conv2d(bi_output, filter=bi_compat_kernel, stride=self.stride, padding=self.padding)

            pairwise = torch.add(ang_output, bi_output)

            outputs = torch.add(unary, pairwise)

            x = outputs
        
        return outputs
