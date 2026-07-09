import torch.nn as nn

class NNBinaryClassifier(nn.Module):
    def __init__(self, in_features, dropout=0.0):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, x):
        return self.seq(x)
