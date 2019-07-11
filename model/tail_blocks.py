import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Classifier(nn.Module):
    def __init__(self, in_dim, fc_dim, num_class):
        super(FC_Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_class)
        self.type = 'fc_cls'

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
