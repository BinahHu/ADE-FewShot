import torch
import torch.nn as nn
from model.tail_blocks import Novel_Classifier, FC_Classifier
from model.component.resnet import resnet18, resnet50
import math
import numpy as np
import random


class ModelBuilder:
    # weight initialization
    def __init__(self, args):
        self.args = args

    def weight_init(self, m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        elif class_name.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.01)

    def build_backbone(self):
        if self.args.architecture == 'resnet18':
            backbone = resnet18()
        elif self.args.architecture == 'resnet50':
            backbone = resnet50()

        backbone.apply(self.weight_init)
        if len(self.args.model_weight) > 0:
            print('Loading weights from {}'.format(self.args.model_weight))
            backbone.load_state_dict(
                torch.load(self.args.model_weight,
                           map_location=lambda storage, loc: storage), strict=False)
        return backbone

    def build_classifier(self):
        classifier = FC_Classifier(self.args, self.args.feat_dim)

        classifier.apply(self.weight_init)
        return classifier
