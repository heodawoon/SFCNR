from __future__ import print_function

import os
import random

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np

import nibabel as nib

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


# Define the function to print and write log
def writelog(file, line):
    file.write(line + '\n')
    print(line)


# Augmentation: sagittal flip
def sagittal_flip(x):
    if random.random() < 0.5:
        x = np.flip(x, axis=0)
    else:
        x = x
    return x


# Augmentation: crop
def random_crop(x):

    sx = random.randint(0, 3)
    sy = random.randint(0, 3)
    sz = random.randint(0, 3)
    x = x[sx:, sy:, sz:]

    return x


# To match size after crop
# cropping + padding = shifting voxels
def match_size(x):
    x = torch.from_numpy(x.copy())
    h1, w1, d1 = x.shape
    h2, w2, d2 = (91, 109, 91)
    while d1 != d2:
        if d1 < d2:
            x = F.pad(x, (0, 1), mode='constant', value=0)
            d1 += 1
        else:
            x = x[:, :, :d2]
            break
    while w1 != w2:
        if w1 < w2:
            x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
            w1 += 1
        else:
            x = x[:, :w2, :]
            break
    while h1 != h2:
        if h1 < h2:
            x = F.pad(x, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
            h1 += 1
        else:
            x = x[:h2, :, :]
            break
    out = x.numpy()
    return out


class JBUH(Dataset):
    def __init__(self, patients_dir, crop_size, age_info, train=True):
        self.patients_dir = patients_dir
        self.age = age_info
        self.train = train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volume_path = os.path.join(patient_dir)
        volume = nib.load(volume_path).get_data()
        volume = self.normlize(volume)
        volume = self.pooling(torch.Tensor(volume)).detach().cpu().numpy()

        if self.train:
            volume = sagittal_flip(volume)
            volume = random_crop(volume)
            volume = match_size(volume)
        volume = np.expand_dims(volume, axis=0)
        volume = torch.from_numpy(volume.astype(float))

        age = self.age[index]
        return (volume, age, patient_dir)

    def pooling(self, volumes):
        m = nn.AvgPool3d(2, stride=2)
        x = m(torch.unsqueeze(volumes, dim=0))
        return x.squeeze()
    def aug_sample(self, volumes):
        """
            Args:
                volumes: list of array, [h, w, d]
            Ret: x, y: [channel, h, w, d]
        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]

        if self.train:
            # crop volume
            x = self.random_crop(x)
            if random.random() < 0.5:
                x = np.flip(x, axis=0)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
        else:
            x = self.center_crop(x)

        return x

    def random_crop(self, x):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume

    def center_crop(self, x):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
