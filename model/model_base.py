import torch
import torch.nn as nn
from model.feature_extractor import LeNet
from model.tail_blocks import FC_Classifier
from model.resnet import resnet18


class ModelBuilder():
    # weight initialization
    def weights_init(self, m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        elif class_name.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.01)

    def build_feature_extractor(self, arch='LeNet', weights=''):
        if arch == 'LeNet':
            feature_extractor = LeNet()
        elif arch == 'resnet18':
            feature_extractor = resnet18()

        feature_extractor.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for feature extractor')
            feature_extractor.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc:storage), strict=False)
        return feature_extractor

    def build_classification_layer(self, args):
        classifier = FC_Classifier(args.feat_dim, 256, args.num_class)
        classifier.apply(self.weights_init)
        return classifier


class LearningModuleBase(nn.Module):
    def __init__(self):
        super(LearningModuleBase, self).__init__()

    def _acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        instance_sum = torch.sum(valid)
        acc = acc_sum.float() / (instance_sum.float() + 1e-10)
        return acc


class LearningModule(LearningModuleBase):
    def __init__(self, feature_extractor, crit, cls=None, seg=None):
        super(LearningModule, self).__init__()
        self.feature_extractor = feature_extractor
        self.cls = cls
        self.seg = seg
        self.crit = crit

    def forward(self, feed_dict, mode='train'):
        feature_map = self.feature_extractor(feed_dict['img_data'])
        acc = 0
        loss = 0
        for crit in self.crit:
            if crit['weight'] == 0:
                continue
            label = feed_dict['{type}_label'.format(type=crit['type'])].long()
            if crit['type'] == 'cls':
                pred = self.cls(feature_map)

            loss += crit['weight'] * crit['crit'](pred, label)
            acc += self._acc(pred, label)

        return loss, acc
