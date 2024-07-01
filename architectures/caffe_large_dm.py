# Caffe architecture for Cifar from
# https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick_train_test.prototxt

import torch.nn as nn

def get_classification_block(dropout, n_classes):
    return Caffe(dropout, n_classes)

class Caffe(nn.Module):
    def __init__(self, dropout, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.dropout = dropout

        self.model = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            #
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.AvgPool2d(kernel_size=3, stride=2),
            #
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.AvgPool2d(kernel_size=3, stride=2),

            # ==================== ATM DM is cut here ======================

            nn.Flatten(),
            nn.Linear(1152, 128),
            nn.Dropout(p=self.dropout),
            # TODO: Output is 64 or 128?
            nn.Linear(128, 64),
            nn.Dropout(p=self.dropout),
            nn.Linear(64, self.n_classes),
        )

    def forward(self, x):
        return self.model(x)

    def __iter__(self):
        return iter(self.model)
