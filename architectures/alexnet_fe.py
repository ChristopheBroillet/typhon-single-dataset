# AlexNet architecture from
# https://en.wikipedia.org/wiki/AlexNet
# https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5

import torch.nn as nn

def get_block(dropout):
    return AlexNet(dropout)

class AlexNet(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        # self.n_classes = n_classes
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=112),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # ==================== ATM DM is cut here ======================

            # nn.Flatten(),
            # nn.Linear(9216, 4096),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            # nn.Dropout(p=self.dropout),
            # nn.Linear(4096, self.n_classes),
            # TODO: possible softmax here?
        )

    def forward(self, x):
        return self.model(x)

    def __iter__(self):
        return iter(self.model)
