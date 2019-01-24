""" Recurrent CRF """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util

class RecurrentCRF( nn.Module ):
    def __init__( self, bilateral_filters, sizes=[3,5], num_iterations=1):
        self.bilateral_filters = bilateral_filters
        self.num_iterations = num_iterations
        self.sizes = sizes

        # model parameter
        self.model_params = []
        # ここもあとでparameterで変える
        self.lider_mask = torch.zeros([BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL, 1],dtype=torch.float32)
    
    def locally_connected_layer(self, inputs, angular_filters, bi_angular_filters, condensing_kernel):
        size_z, size_a = self.sizes
        pad_z, pas_a = size_z//2, size_a//2
        half_filter_dim = (size_z*size_a)//2
        batch, zenith, azimuth, in_channel = list(inputs.size())
        
        ang_output = F.conv2d(inputs, filter=angular_filters, stride=1, padding=0)

        bi_ang_output = F.conv2d(inputs, filter=bi_angular_filters, stride=1, padding=0)

        condensed_input = F.conv2d(
                inputs*self.lider_mask, filter=condensing_kernel, stride=1, padding=0
        )
        condensed_input = condensed_input.view(
                batch, zenith, azimuth, size_z*size_a-1, in_channel
        )

        bi_output = 

    def forward( self, x ):
        # initialize compatibilty matrices
        compat_kernel_init = torch.from_numpy(
                np.reshape(
                    np.ones((NUM_CLASS, NUM_CLASS), dtype="float32") - np.identity(NUM_CLASS, dtype="float32"),
                    [1, 1, NUM_CLASS, NUM_CLASS]
                )
        )

        angular_compat_kernel = compat_kernel_init * ANG_FILTER_CONF
        angular_compat_kernel.requires_grad_()

        self.model_params += [bi_compat_kernal, angular_compat_kernel]

        condensing_kernel = torch.from_numpy(
                util.condensing_matrix(sizes[0], sizes[1], NUM_CLASS)
        )

        angular_filters = torch.from_numpy(
                util.angular_filter_kernel(sizes[0], sizes[1], NUM_CLASS, ANG_THETA_A**2)
        )

        bi_angular_filters = torch.from_numpy(
                util.angular_filter_kernel(sizes[0], sizes[1], NUM_CLASS, BILATERAL_THETA_A**2)
        )

        for it in range(self.num_iterations):
            unary = F.softmax(x, dim=-1)


