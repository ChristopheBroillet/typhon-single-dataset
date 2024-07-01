import torch.nn as nn

def get_classification_block(dropout, n_classes):
    return nn.Sequential(

        nn.Linear(256, 64),
        nn.ELU(),

        nn.Dropout(p=dropout),

        nn.Linear(64, 16),
        nn.ELU(),

        nn.Linear(16, n_classes)
    )
