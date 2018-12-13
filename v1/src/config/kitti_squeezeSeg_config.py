"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def kitti_squeezeSeg_config():
	"""Specify the parameters to tune below."""
    mc = base_model_config('KITTI')
