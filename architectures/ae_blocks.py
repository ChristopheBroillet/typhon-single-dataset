import torch
import torch.nn as nn


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Disable bias for convolutions direclty followed by a batch norm
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def __iter__(self):
        return iter([self.conv, self.bn, self.relu])


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, padding=1)
        self.relu = nn.ELU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.relu(x)
        return x

    def __iter__(self):
        return iter([self.upconv, self.relu])
