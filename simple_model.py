import torch.nn as nn
import torch.nn.functional as F


class simple_model(nn.Module):
    def __init__(self, num_classes=10):
        super(simple_model, self).__init__()
        self.fc1 = nn.Linear(28*28, 1000)
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.fc2 = nn.Linear(1000, 500)
        self.bn2 = nn.BatchNorm1d(num_features=500)
        self.fc3 = nn.Linear(500, num_classes)

    def forward(self, x):
        hidden_1 = F.relu(self.bn1(self.fc1(x)))
        hidden_2 = F.relu(self.bn2(self.fc2(hidden_1)))
        hidden_3 = self.fc3(hidden_2)
        out = F.log_softmax(hidden_3, dim=1)
        return out

