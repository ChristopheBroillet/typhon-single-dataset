# AlexNet architecture from
# https://en.wikipedia.org/wiki/AlexNet
# https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5

import torch.nn as nn

def get_classification_block(dropout, n_classes):
    return AlexNet(dropout, n_classes)

class AlexNet(nn.Module):
    def __init__(self, dropout, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, self.n_classes),
        )

    def forward(self, x):
        return self.model(x)

    def __iter__(self):
        return iter(self.model)
