import random
import numpy as np

import torch
import torch.nn.functional as F

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

