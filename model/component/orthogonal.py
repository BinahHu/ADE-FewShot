import torch.nn as nn
import torch
import numpy as np
from model.component.normalize import Normalize


class Orthogonal(nn.Module):
    def __init__(self):
        super(Orthogonal, self).__init__()
        self.split = 4
        self.norm = Normalize()

    def forward(self, x):
        num, dim = x.shape
        parts = []
        for i in range(self.split):
            down = i * int(dim / self.split)
            up = (i + 1) * int(dim / self.split)
            parts.append(x[:, down:up])

        indices = np.arange(self.split)
        np.random.shuffle(indices)
        group = int(self.split / 2)

        loss_sum = torch.zeros(num, num).cuda()
        for i in range(group):
            loss_sum += torch.mm(self.norm(parts[indices[2*i]]), self.norm(parts[indices[2*i+1]]).t())

        return loss_sum.mean()
