""" Kitti Dataset for SqueezeSeg """

import pandas as pd

import numpy as np

import torch
from torch.utils.data import Dataset

import os
import sys

class KittiDataset(Dataset):
    def __init__(self, mc, csv_file, root_dir, transform=None):
        self.mc = mc

        self.lidar_2d_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.lidar_2d_csv)

    def __getitem__(self, idx):
        mc = self.mc

        lidar_name = os.path.join(self.root_dir, self.lidar_2d_csv.iloc[idx, 0])

        lidar_data = np.load(lidar_name).astype(np.float32)

        if mc.DATA_AUGMENTATION:
            if mc.RANDOM_FLIPPING:
                if np.random.rand() > 0.5:
                    # flip y
                    lidar_data = lidar_data[:, ::-1, :]
                    lidar_data[:, :, 1] *= -1

        lidar_inputs =  lidar_data[:, :, :5] # x, y, z, intensity, depth(range)

        lidar_mask = np.reshape(
            (lidar_inputs[:, :, 4] > 0) * 1,
            [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
        )

        # Normalize Inputs
        lidar_inputs = (lidar_inputs - mc.INPUT_MEAN) / mc.INPUT_STD

        lidar_label = lidar_data[:, :, 5]

        weight = np.zeros(lidar_label.shape)
        for l in range(mc.NUM_CLASS):
            weight[lidar_label == l] = mc.CLS_LOSS_WEIGHT[int(l)]

        if self.transform:
            lidar_inputs = self.transform(lidar_inputs)
            lidar_mask = self.transform(lidar_mask)

        return (lidar_inputs.float(), lidar_mask.float(), torch.from_numpy(lidar_label.copy()).long(), torch.from_numpy(weight.copy()).float())
