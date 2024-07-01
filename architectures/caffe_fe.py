# Caffe architecture for Cifar from
# https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick_train_test.prototxt

import torch.nn as nn

def get_block(dropout):
    return Caffe(dropout)

class Caffe(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        # self.n_classes = n_classes
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.AvgPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.AvgPool2d(kernel_size=3, stride=2),

            # ==================== ATM DM is cut here ======================

            # nn.Flatten(),
            # nn.Linear(576, 64),
            # nn.Linear(64, self.n_classes),
        )

    def forward(self, x):
        return self.model(x)

    def __iter__(self):
        return iter(self.model)
