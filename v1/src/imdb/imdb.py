"""The data base wrapper class"""

import os
import random
import shutil

import numpy as np

from utils.util import *

class imdb(object):
    """Image database"""

    def __init__(self, name, mc):
        self._name = name
        self._image_set = []
        self._image_index = []
        self._data_root_path = []
        self.mc = mc

        # batch reader
        self._perm_idx = []
        self._cur_idx = 0

    @property
    def name(self):
        return self._name

    @property
    def image_idx(self):
        return self._image_idx

    @property
    def image_set(self):
        return self.image_set

