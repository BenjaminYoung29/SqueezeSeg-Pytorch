""" Recurrent CRF """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import util

class RecurrentCRF( nn.Module ):
    def __init__( self, bilateral_filters, sizes=[3,5], num_iterations=1):
        self.bilateral_filters = bilateral_filters
        
        # model parameter
        self.model_params = []

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

        for it in range(num_iterations):
            unary = 
