import os
import random
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class UK_Dataset_train_loader(Dataset):

    def __init__(self, args, X, MRI, CSV, transform=True):
        self.X = X
        self.MRI = MRI
        self.csv = CSV
        self.transform = transform
        self.args = args

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        index = self.X[idx]
        csv = self.csv
        concat_img = self.MRI
        image = concat_img[index]

        h, w, d = image.shape
        img = torch.from_numpy(image).float().view(1, h, w, d)

        age = csv['first_age_condition'].iloc[index]

        if self.transform is True:
            image_aug = sagittal_flip(image)
            image_aug = random_crop(image_aug)
            image_aug = match_size(image_aug)

            h, w, d = image_aug.shape
            img_aug = torch.from_numpy(image_aug).float().view(1, h, w, d)
            train = TensorDataset(img, img_aug, age)
            train_dataloader = DataLoader(train, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            return train_dataloader
                    
        train = TensorDataset(img, age)
        train_dataloader = DataLoader(train, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        
        return train_dataloader


class UK_Dataset_valid_loader(Dataset):

    def __init__(self, X, MRI, CSV):
        self.X = X
        self.MRI = MRI
        self.csv = CSV

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        index = self.X[idx]
        csv = self.csv
        concat_img = self.MRI
        image = concat_img[index]

        h, w, d = image.shape
        img = torch.from_numpy(image).float().view(1, h, w, d)

        age = csv['first_age_condition'].iloc[index]
                    
        valid = TensorDataset(img, age)
        valid_dataloader = DataLoader(valid, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        
        return valid_dataloader



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
