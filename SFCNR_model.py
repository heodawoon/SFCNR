'''
Accurate brain age prediction with lightweight deep neural networks Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith Medical Image Analysis (2021); doi: https://doi.org/10.1016/j.media.2020.101871

Peng, H. et al., (2021). UKBiobank_deep_pretrain [Source code]. GitHub. https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain
'''

import torch
import torch.nn as nn

class SFCNR(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=1):
        super(SFCNR, self).__init__()

        # The SFCN-based convolutional block
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        # The regression block
        self.reg_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*2*3*2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim))
    
    # The convolution layer
    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU())
        return layer

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x_c = self.reg_block(x_f)
        out = x_c
        return out
