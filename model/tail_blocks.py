import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Classifier(nn.Module):
    def __init__(self, args, in_dim, fc_dim):
        super(FC_Classifier, self).__init__()
        self.num_class = args.num_class
        self.down_sampling_rate = args.down_sampling_rate
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_dim, self.num_class)
        self.type = 'fc_cls'

    def forward(self, x):
        feature_map, scale, anchors, anchor_num = x
        anchors = anchors * scale / self.down_sampling_rate
        anchors.detach()
        scale.detach()
        anchors.int()

        pred = torch.zeros(anchor_num, self.num_class)
        for i in range(anchor_num):
            anchor = anchors[i]
            feature = feature_map[:, anchor[0][1]:anchor[1][1], anchor[0][0]:anchor[1][0]]
            pred[i, :] = self.fc1(self.global_pool(feature))
        return pred
