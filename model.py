import torch.nn as nn

class NNBinaryClassifier(nn.Module):
    def __init__(self, in_features, hidden_dims=(128, 64), dropout=0.0):
        super().__init__()

        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_features=prev, out_features=h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev = h

        layers.append(nn.Linear(in_features=prev, out_features=1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
