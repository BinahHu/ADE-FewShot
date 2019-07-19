import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Classifier(nn.Module):
    def __init__(self, args, in_dim):
        super(FC_Classifier, self).__init__()
        self.num_class = args.num_class
        self.down_sampling_rate = args.segm_downsampling_rate
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, self.num_class)
        self.type = 'fc_cls'

    def forward(self, x):
        feature_map, scale, anchors, anchor_num = x
        anchors = anchors.detach().cpu()
        scale = scale.detach().cpu()
        anchors = anchors / self.down_sampling_rate
        anchor_num = int(anchor_num)

        pred = torch.zeros(anchor_num, self.num_class)
        for i in range(anchor_num):
            anchor = anchors[i]
            anchor[2] = anchor[2] * scale[0]
            anchor[3] = anchor[3] * scale[0]
            anchor[0] = anchor[0] * scale[1]
            anchor[1] = anchor[1] * scale[1]
            anchor = anchor.int()
            if anchor[2] == anchor[3]:
                if anchor[2] != 0:
                    anchor[2] -= 1
                else:
                    anchor[3] += 1
            if anchor[0] == anchor[1]:
                if anchor[0] != 0:
                    anchor[0] -= 1
                else:
                    anchor[1] += 1
            feature = feature_map[:, anchor[2]:anchor[3], anchor[0]:anchor[1]].squeeze(0)
            feature = self.global_pool(feature)
            feature = feature.view(1, self.in_dim)
            pred[i, :] = self.fc1(feature)
        return pred
