import torch
import torch.nn as nn
from model.component.classifier import Classifier
from model.component.resnet import resnet18, resnet34, resnet10
from model.component.attr import AttrClassifier
from model.component.seg import BinaryMaskPredictor


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
        elif self.args.architecture == 'resnet10':
            backbone = resnet10()
        elif self.args.architecture == 'resnet34':
            backbone = resnet34()

        backbone.apply(self.weight_init)
        return backbone

    def build_classifier(self):
        classifier = Classifier(self.args)
        classifier.apply(self.weight_init)
        return classifier

    def build_attr(self):
        attr_classifier = AttrClassifier(self.args)
        attr_classifier.apply(self.weight_init)
        return attr_classifier

    def build_seg(self):
        segment_module = BinaryMaskPredictor(self.args)
        segment_module.apply(self.weight_init)
        return segment_module
