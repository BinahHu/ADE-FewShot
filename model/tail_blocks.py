import torch
import torch.nn as nn
import torch.nn.functional as F
from roi_align.roi_align import  RoIAlign
from torch.autograd import Variable
import numpy as np


"""
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
"""

def to_variable(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class FC_Classifier(nn.Module):
    def __init__(self, args, in_dim):
        super(FC_Classifier, self).__init__()
        self.crop_height = int(args.crop_height)
        self.crop_width = args.crop_width
        self.roi_align = RoIAlign(args.crop_height, args.crop_width, transform_fpcoor=True)
        self.num_class = args.num_class
        self.down_sampling_rate = args.segm_downsampling_rate
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim * self.crop_width * self.crop_height, self.num_class)
        self.type = 'fc_cls'
        self.output = 'dumb'

    def forward(self, x):
        feature_map, scale, anchors, anchor_num = x
        anchors = np.array(anchors.detach().cpu())
        scale = np.array(scale.detach().cpu())
        anchors = anchors / self.down_sampling_rate
        anchor_num = int(anchor_num)
        anchors[:, 2] = anchors[:, 2] * scale[0]
        anchors[:, 3] = anchors[:, 3] * scale[0]
        anchors[:, 0] = anchors[:, 0] * scale[1]
        anchors[:, 1] = anchors[:, 1] * scale[1]
        anchors[:, [1, 2]] = anchors[:, [2, 1]]
        anchor_index = np.zeros(anchor_num)
        anchor_index = to_variable(anchor_index).int()
        anchors = to_variable(anchors[:anchor_num, :]).float()
        pred = torch.zeros(anchor_num, self.num_class)
        feature_map = feature_map.unsqueeze(0)
        # print('feature map size {}'.format(feature_map.shape))
        feature = self.roi_align(feature_map, anchors, anchor_index)
        # print('Feature size is {}'.format(feature.shape))

        if self.output == 'feat':
            return feature

        feature = feature.view(-1, self.in_dim * self.crop_height * self.crop_width)
        # print('Feature shape after view is {}'.format(feature.shape))
        pred = self.fc1(feature)
        # print('pred shape {}\n'.format(pred.shape))
        return pred
