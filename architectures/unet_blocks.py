import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Disable bias for convolutions direclty followed by a batch norm
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1,  bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv(x)
        output = self.pool(x)
        # Return one single value
        # Output and skip (x)
        return [output, x]


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, x):
        # Split
        x, skip = x
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
