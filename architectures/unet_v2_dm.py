# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
import torch
import torch.nn as nn
from architectures.unet_blocks import conv_block, encoder_block, decoder_block


def get_block(dropout, in_channels=1):
    return Unet_container(in_channels=in_channels)


class Unet_container(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # """ Encoder """
        # self.e1 = encoder_block(in_channels, 64)
        # self.e2 = encoder_block(64, 128)
        # self.e3 = encoder_block(128, 256)
        # self.e4 = encoder_block(256, 512)
        #
        # """ Bottleneck """
        # self.b = conv_block(512, 1024)
        #
        # """ Decoder """
        # self.d1 = decoder_block(1024, 512)
        # self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # """ Encoder """
        # x, s1 = self.e1(x)
        # x, s2 = self.e2(x)
        # x, s3 = self.e3(x)
        # x, s4 = self.e4(x)
        # """ Bottleneck """
        # x = self.b(x)
        #
        # """ Decoder """
        # x = self.d1([x, s4])
        # x = self.d2([x, s3])
        x, s2, s1 = inputs
        x = self.d3([x, s2])
        x = self.d4([x, s1])
        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x

    def __iter__(self):
        return iter([self.d3, self.d4, self.final_conv, self.sigmoid])
