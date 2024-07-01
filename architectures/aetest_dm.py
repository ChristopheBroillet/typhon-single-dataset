# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
import torch
import torch.nn as nn
from architectures.ae_blocks import encoder_block, decoder_block


def get_block(dropout, in_channels=1):
    return AE_container(in_channels=in_channels)


class AE_container(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        """ Decoder """
        self.d1 = decoder_block(128, 64)
        self.d2 = decoder_block(64, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        # x = self.sigmoid(x)
        return x

    def __iter__(self):
        return iter([self.d1, self.d2, self.sigmoid])
