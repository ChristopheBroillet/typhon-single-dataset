# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
import torch
import torch.nn as nn
from architectures.ae_blocks import encoder_block, decoder_block


def get_block(dropout, in_channels=1):
    return AE_container(in_channels=in_channels)


class AE_container(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(in_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(x)
        # x = self.pool(x)
        return x

    def __iter__(self):
        return iter([self.e1, self.e2, self.pool])
