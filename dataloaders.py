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
