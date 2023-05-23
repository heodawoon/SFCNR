import os
import glob
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import warnings
warnings.filterwarnings('ignore')

import torch.backends.cudnn as cudnn

from utils import writelog
from SFCN_model import SFCNR


# Fix a random seed for reproduction
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--outer_fold', type=int, default=1)
parser.add_argument('--inner_fold', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--optype', type=str, default='adam')
parser.add_argument('--finelr', type=float, default=1e-5)
parser.add_argument('--finelamb2', type=float, default=1e-6)
parser.add_argument('--earlystop', type=int, default=3)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint_pth = './SFCNR_MAE_2.623_epoch_77.pt'
#checkpoint_pth = '/DataCommon3/daheo/JBUH/result/Ageprediction/pretrain/20220923.22.10.20_regress_age_norm_epoch_300_bsize_8_opt_type_sgd_lr_0.01_wdecay_0.001_output_dim_1_earlystop_30/model/SFCN_MAE_2.623_epoch_77.pt'

date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
directory = os.path.join('./finetuning', 'Outer_Fold'+str(args.outer_fold),
                         'Inner_Fold'+str(args.inner_fold), 'Batch_size_'+str(args.batch_size), 'Optype_'+args.optype, 'lr'+str(args.finelr),
                         date_str + '_epoch_' + str(args.epoch) + '_finelamb2_' + str(args.finelamb2) + '_earlystop'+str(args.earlystop))

ckpoint_dir = os.path.join(directory, 'checkpoint')
log_pth = os.path.join(directory, 'log')
model_pth = os.path.join(directory, 'model')

if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(ckpoint_dir)
    os.makedirs(log_pth)
    os.makedirs(model_pth)
    os.makedirs(os.path.join(directory, 'tflog'))

tfw_train = tf.compat.v1.summary.FileWriter(os.path.join(directory, 'tflog', 'train_'))
tfw_valid = tf.compat.v1.summary.FileWriter(os.path.join(directory, 'tflog', 'valid_'))


# Text Logging
f = open(os.path.join(directory, 'setting.log'), 'a')
writelog(f, '======================')
writelog(f, 'Date: %s' % date_str)
writelog(f, 'Outer Fold: %d' % args.outer_fold)
writelog(f, 'Inner Fold: %d' % args.inner_fold)
writelog(f, '======================')
writelog(f, 'Epoch: %d' % args.epoch)
writelog(f, 'Batch Size: %d' % args.batch_size)
writelog(f, '======================')
writelog(f, 'Opt type: %s' % args.optype)
writelog(f, 'Learning Rate: %s' % str(args.finelr))
writelog(f, 'Weight Decay: %s' % str(args.finelamb2))
writelog(f, 'Earlystop: %d' % args.earlystop)
writelog(f, '======================')
f.close()

f = open(os.path.join(directory, 'log.log'), 'a')

writer = SummaryWriter(log_pth)

# Load Dataset
path_train_images = './Dataset/Combat_MNI_space/After_combat/HC_tr_combat_nii'
# path_valid_images = './Combat_MNI_space/After_combat/HC_tr_combat_nii'
# path_test_images = './Combat_MNI_space/After_combat/HC_te_combat_nii'

#path_train_images = '/DataCommon3/daheo/JBUH/Dataset/Combat_MNI_space/After_combat/HC_tr_combat_nii'
# path_valid_images = '/DataCommon3/daheo/JBUH/Dataset/Combat_MNI_space/After_combat/HC_tr_combat_nii'
# path_test_images = '/DataCommon3/daheo/JBUH/Dataset/Combat_MNI_space/After_combat/HC_te_combat_nii'

outer_index_folder_pth = os.path.join('/DataCommon3/daheo/JBUH/Dataset/Combat_MNI_space/After_combat/InnerLoop_NestedCV_Folds_new_221007', 'Outer_Fold'+str(args.outer_fold))
inner_index_folder_pth = os.path.join('/DataCommon3/daheo/JBUH/Dataset/Combat_MNI_space/After_combat/InnerLoop_NestedCV_Folds_new_221007',
                                      'Outer_Fold'+str(args.outer_fold), 'Inner_Fold'+str(args.inner_fold))

inner_tr_id_age = np.load(os.path.join(inner_index_folder_pth, 'Out_Fold'+str(args.outer_fold)+'_In_Fold'+str(args.inner_fold)+'_train.npz'))
inner_val_id_age = np.load(os.path.join(inner_index_folder_pth, 'Out_Fold'+str(args.outer_fold)+'_In_Fold'+str(args.inner_fold)+'_valid.npz'))
outer_te_id_age = np.load(os.path.join(outer_index_folder_pth, 'Outer_Fold'+str(args.outer_fold)+'_test.npz'))


# Inner loop train id and age information
inner_tr_id = inner_tr_id_age['ID']
inner_tr_age = inner_tr_id_age['Age']
mean_age_ = inner_tr_age.mean()
std_age_ = inner_tr_age.std()

# Inner loop valid id and age information
inner_val_id = inner_val_id_age['ID']
inner_val_age = inner_val_id_age['Age']


tr_img_path = []
tr_age = []
for fidx in range(len(inner_tr_id)):
    tr_img_path.append(os.path.join(path_tr_images, inner_tr_id[fidx] + '_flirt_restore.nii.gz'))
    tr_age.append(inner_tr_age[fidx])

val_img_path = []
val_age = []
for fidx in range(len(inner_val_id)):
    val_img_path.append(os.path.join(path_tr_images, inner_val_id[fidx] + '_flirt_restore.nii.gz'))
    val_age.append(inner_val_age[fidx])

# Define Loaders
trainloader = JBUH(tr_img_path, crop_size=(91, 109, 91), age_info=tr_age, train=True)
tr_dataloader = DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, drop_last=True)
validloader = JBUH(val_img_path, crop_size=(91, 109, 91), age_info=val_age, train=False)
val_dataloader = DataLoader(validloader, batch_size=args.batch_size, shuffle=False, drop_last=False)

dataloaders = {'train': tr_dataloader,
               'valid': val_dataloader
               }

# Initialize Generator and Discriminator
model = SFCNR().to(device)
pretrained_dict = torch.load(checkpoint_pth)
model.load_state_dict(pretrained_dict)

# Optimizers
if args.optype == 'adam':
    optimizer_fine = torch.optim.Adam(model.parameters(), lr=args.finelr, weight_decay=args.finelamb2)
elif args.optype == 'sgd':
    optimizer_fine = torch.optim.SGD(model.parameters(), lr=args.finelr, weight_decay=args.finelamb2)

# Loss function
mae_loss = nn.L1Loss().cuda()

def age_MAE(preds, targets):
    running_mae = torch.abs(preds - targets).sum().data
    return running_mae

# TRAIN function
def train(dataloader, epoch, dir='.'):
    model.train()
    train_loss = 0
    train_mae = 0.0
    train_losses = []
    train_mae_losses = []
    for batch, input_data in enumerate(tqdm(dataloader)):
        optimizer_fine.zero_grad()  # reset gradient

        input_img, input_age, _ = input_data
        input_img = Variable(input_img.type(torch.FloatTensor)).cuda()
        input_age_norm = (input_age.clone() - torch.Tensor(np.tile(mean_age_, input_age.shape))) / torch.Tensor(np.tile(std_age_, input_age.shape))
        input_age_norm = Variable(input_age_norm).squeeze().cuda()

        # forward
        output = model(input_img)
        loss = mae_loss(output.squeeze(), input_age_norm)

        # backward
        loss.backward()
        optimizer_fine.step()

        # step the learning rate
        train_loss += (loss.item()*input_img.shape[0])

        # evaluation metrics
        output_x = (output.squeeze() * torch.Tensor(np.tile(std_age_, output.squeeze().shape)).to(device)) + torch.Tensor(np.tile(mean_age_, output.squeeze().shape)).to(device)
        MAE_train = age_MAE(output_x.squeeze(), input_age.to(device))
        train_mae += (MAE_train.item()*input_img.shape[0])
    
    # print the loss value
    Train_Loss = train_loss / len(dataloader.dataset)
    Train_MAE = train_mae / len(dataloader.dataset)
    train_losses.append(Train_Loss)
    train_mae_losses.append(Train_MAE)

    print('[Epoch '+str(epoch)+'/'+str(args.epoch)+']  Train loss = '+str(Train_Loss)[0:7]+',  Train MAE = '+str(Train_MAE)[0:7])
    writer.add_scalar('/train/loss', Train_Loss, global_step=epoch)

    # Tensorboard Logging
    info = {
            'loss': Train_Loss,
            'MAE_real_age': Train_MAE,
           }
    for tag, value in info.items():
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        tfw_train.add_summary(summary, epoch)

    return train_losses, train_mae


# EVALUATION function
def evaluate(phase, dataloader, epoch, dir='.'):
    model.eval()
    valid_loss = 0
    valid_mae = 0.0
    valid_losses = []
    valid_mae_losses = []
    
    with torch.no_grad():
        for batch, input_data in enumerate(tqdm(dataloader)):

            input_img, input_age, _ = input_data
            input_img = Variable(input_img.type(torch.FloatTensor)).cuda()
            input_age_norm = (input_age.clone() - torch.Tensor(np.tile(mean_age_, input_age.shape))) / torch.Tensor(np.tile(std_age_, input_age.shape))
            input_age_norm = Variable(input_age_norm).squeeze().cuda()

            output = model(input_img)
            loss = mae_loss(output.squeeze(), input_age_norm)

            valid_loss += (loss.item()*input_img.shape[0])

            # evaluation metrics
            output_x = (output.squeeze() * torch.Tensor(np.tile(std_age_, output.squeeze().shape)).to(device)) + torch.Tensor(np.tile(mean_age_, output.squeeze().shape)).to(device)
            MAE_valid = age_MAE(output_x.squeeze(), input_age.to(device))
            valid_mae += (MAE_valid.item()*input_img.shape[0])

        # print the loss value
        Valid_Loss = valid_loss / len(dataloader.dataset)
        Valid_MAE = valid_mae / len(dataloader.dataset)
        valid_losses.append(Valid_Loss)
        valid_mae_losses.append(Valid_MAE)
        print('[Epoch ' + str(epoch) + '/' + str(args.epoch) + ']  Valid loss = ' + str(Valid_Loss)[0:7] + ',  Valid MAE = ' + str(Valid_MAE)[0:7])

        writer.add_scalar('/'+phase+'_test/loss', Valid_Loss, global_step=epoch)


        # Tensorboard Logging
        info = {
                'loss': Valid_Loss,
                'MAE_real_age': Valid_MAE,
               }

        for tag, value in info.items():
            summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
            if phase == 'Valid':
                tfw_valid.add_summary(summary, epoch)

    return Valid_Loss, valid_losses, valid_mae_losses


# Best epoch checking
valid = {
    'epoch': 0,
    'loss': np.Inf,
}

for epoch in range(args.epoch):

    _, _ = train(dataloaders['train'], epoch, dir=directory)
    loss_val, _, _ = evaluate('Valid', dataloaders['valid'], epoch, dir=directory)

    if loss_val < valid['loss']:
        # Saving models
        torch.save(model.state_dict(), model_pth + '/model_epoch_{}.pt'.format(valid['epoch']))
        torch.save(model.state_dict(), os.path.join(model_pth, 'Best_model.pt'))

        print(" ========== Saving model ")

        writelog(f, 'Best validation loss is found! Validation loss : %f' % loss_val)
        writelog(f, 'Models at Epoch %d are saved!' % epoch)
        valid['loss'] = loss_val
        valid['epoch'] = epoch
