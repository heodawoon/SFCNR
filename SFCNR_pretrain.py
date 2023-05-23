import os
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split

from SFCNR_model import SFCNR
from utils import writelog
from dataloaders import UK_Dataset_train, UK_Dataset_valid

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default="4")
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--opt_type', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--wdecay', type=float, default=1e-3)
parser.add_argument('--outdim', type=int, default=1)
parser.add_argument('--earlystop', type=int, default=30)
# there a rule of thumb to make it 10% of number of epoch.
# https://medium.com/zero-equals-false/early-stopping-to-avoid-overfitting-in-neural-network-keras-b68c96ed05d9
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
directory = './result/Ageprediction/pretrain/' + date_str + '_regress_age_norm_epoch_' + str(
    args.epoch) + '_bsize_' + str(
    args.batch_size) + '_opt_type_' + args.opt_type + '_lr_' + str(args.lr) + '_wdecay_' + str(
    args.wdecay) + '_output_dim_' + str(args.outdim) + '_earlystop_' + str(args.earlystop)

ckpoint_dir = os.path.join(directory, 'checkpoint')
model_dir = os.path.join(directory, 'model')
logdir = os.path.join(directory, 'log')
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(ckpoint_dir)
    os.makedirs(model_dir)
    os.makedirs(logdir)

f = open(os.path.join(directory, 'setting.log'), 'a')
writelog(f, '======================')
writelog(f, 'Gpu ID: %s' % args.gpu_id)
writelog(f, 'Optimizer Type: %s' % args.opt_type)
writelog(f, 'Learning Rate: %s' % str(args.lr))
writelog(f, 'Weight Decay: %s' % str(args.wdecay))
writelog(f, 'Batch Size: %d' % args.batch_size)
writelog(f, 'Out Dim: %d' % args.outdim)
writelog(f, 'Epoch: %d' % args.epoch)
writelog(f, '======================')
f.close()

writer = SummaryWriter(log_dir=logdir)

# Total Data path (Training, validation(MAE / PAD)
# train_csv = pd.read_csv("./UK_Biobank/first_visit_final_7590.csv")
# train_npy = np.load("./UK_Biobank/first_visit_final_7590.npz", mmap_mode="r")['data']

train_csv = pd.read_csv('./UK_Biobank/first_visit_final_7590.csv')
train_npy = \
np.load("./UK_Biobank/3D_image_npy/Train_7590_AvpFirst.npz", mmap_mode="r")['data']

train_idx, tmp_idx = train_test_split(train_csv['index'].values, test_size=0.2, random_state=7)
valid_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=7)

train_age_tmp = train_csv['first_age_condition'].iloc[train_idx]
train_age_mean = train_age_tmp.mean()
train_age_std = train_age_tmp.std()

# Dataloader
train = UK_Dataset_train(train_idx, train_npy, train_csv, transform=True)
train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
valid = UK_Dataset_valid(valid_idx, train_npy, train_csv)
valid_dataloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, drop_last=False)

dataloaders = {'train': train_dataloader,
               'valid': valid_dataloader,}

# Loss function
mae_loss = nn.L1Loss().cuda()
model = SFCNR().to(device)

# Optimizers & Scheduler
if args.opt_type == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.opt_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)


def training(epoch, train_dataloader):

    # Training
    model.train()

    train_loss = 0
    train_mae = 0

    for batch, input_data in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()  # reset gradient
        _, input_img, input_age = input_data
        bs = input_img.shape[0]

        input_img = Variable(input_img).cuda()
        input_age_norm = (input_age.clone() - train_age_mean) / train_age_std
        input_age_norm = Variable(input_age_norm)

        # forward
        output = model(input_img)
        loss = mae_loss(output.squeeze().cuda(), input_age_norm.squeeze().cuda()).cuda()

        # backward
        loss.backward()
        # update weight
        optimizer.step()

        train_loss += loss.item()

        # evaluation metrics
        output_x = (output.squeeze().detach().cpu().numpy() * train_age_std) + train_age_mean
        MAE_train = np.abs(output_x - input_age.squeeze().detach().cpu().numpy()).sum() / bs
        train_mae += MAE_train

    scheduler.step()

    # calculate mean for each epoch
    Train_Loss = train_loss / len(train_dataloader)
    Train_MAE = train_mae / len(train_dataloader)

    print("Train Loss: {:.3f}".format(Train_Loss),
          "Train MAE: {:.3f}".format(Train_MAE))
    writer.add_scalar('/Train/loss', (train_loss / len(train_dataloader)), global_step=epoch)
    writer.add_scalar('/Train/MAE', (train_mae / len(train_dataloader)), global_step=epoch)

def evaluation(epoch, phase, valid_dataloader):

    # Validation
    model.eval()
    valid_loss = 0
    valid_mae = 0

    # validation loop
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_dataloader)):
            images, labels = data
            val_bs = images.shape[0]
            images = Variable(images).cuda()
            labels_norm = (labels - train_age_mean) / train_age_std
            labels_norm = labels_norm.cuda()

            # generate outputs through model
            output_valid = model(images)

            # calcuate loss
            loss = mae_loss(output_valid.squeeze().cuda(), labels_norm.squeeze().cuda()).cuda()
            valid_loss += loss.item()

            # evalutation metrics
            x_valid = (output_valid.squeeze().detach().cpu().numpy() * train_age_std) + train_age_mean
            MAE_valid = np.abs(x_valid - labels.detach().cpu().numpy()).sum() / val_bs
            valid_mae += MAE_valid

    # calculate mean for each epoch
    Valid_Loss = valid_loss / len(valid_dataloader)
    Valid_MAE = valid_mae / len(valid_dataloader)

    print(phase + " Loss: {:.3f}".format(Valid_Loss),
          phase + " MAE: {:.3f}".format(Valid_MAE), )
    writer.add_scalar('/' + phase + '/loss', (valid_loss / len(valid_dataloader)), global_step=epoch)
    writer.add_scalar('/' + phase + '/MAE', (valid_mae / len(valid_dataloader)), global_step=epoch)

    return valid_loss, valid_mae



min_loss = np.inf
not_improve = 0
for epoch in range(args.epoch):
    training(epoch, dataloaders['train'])
    valid_loss, valid_mae = evaluation(epoch, 'Valid', dataloaders['valid'])

    if min_loss > (valid_loss / len(valid_dataloader)):
        print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (valid_loss / len(valid_dataloader))))
        min_loss = (valid_loss / len(valid_dataloader))
        print('saveing model...')
        torch.save(model.state_dict(),
                   model_dir + '/SFCNR_MAE_{:.3f}_epoch_{}.pt'.format((valid_mae / len(valid_dataloader)), epoch))
        not_improve = 0
    else:
        not_improve += 1
        print(f'Loss Not Decrease for {not_improve} time')
        if not_improve == args.earlystop:
            print('Loss not decrease for {} times, Stop Training'.format(args.earlystop))
            break

