"""Model configuration for pascal dataset"""

import numpy as np

from .config import base_model_config

def kitti_squeezeSeg_config():
    """Specify the parameters to tune below."""
    mc = base_model_config('KITTI')

    mc.CLASSES = ['unkwon', 'car', 'pedestrian', 'cyclist']
    mc.NUM_CLASS = len(mc.CLASSES)
    mc.CLS_2_ID = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
    mc.CLS_LOSS_WEIGHT = np.array([1/15.0, 1.0, 10.0, 10.0])
    mc.CLS_COLOR_MAP = np.array([[0.00, 0.00, 0.00],
                                 [0.12, 0.56, 0.37],
                                 [0.66, 0.55, 0.71],
                                 [0.58, 0.72, 0.88]])

    mc.AZIMUTH_LEVEL = 512
    mc.ZENITH_LEVEL = 64

    mc.LCN_HEIGHT = 3
    mc.LCN_WIDTH = 5
    mc.RCRF_ITER = 3
    mc.BILATERAL_THETA_A = np.array([.9, .9, .6, .6])
    mc.BILATERAL_THETA_R = np.array([.015, .015, .01, .01])
    mc.BI_FILTER_COEF = 0.1
    mc.ANG_THETA_A = np.array([.9, .9, .6, .6])
    mc.ANG_FILTER_COEF = 0.02

    mc.CLS_LOSS_COEF = 15.0

    mc.DATA_AUGMENTATION = True
    mc.RANDOM_FLIPPING = True

    # x, y, z, intensity, distance for Normalization
    mc.INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
    mc.INPUT_STD = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

    return mc
