import torch
import torch.nn as nn
from model.feature_extractor import LeNet
from model.tail_blocks import Novel_Classifier, FC_Classifier
from model.resnet import resnet18
import math
import numpy as np
import random


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
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return feature_extractor

    def build_classification_layer(self, args):
        if args.cls == 'linear':
            classifier = FC_Classifier(args, args.feat_dim)
        elif args.cls == 'novel_cls':
            classifier = Novel_Classifier(args.feat_dim * args.crop_height * args.crop_width, args.num_class)
        elif args.cls == 'linear2':
            classifier = FC_Classifier2(args.feat_dim, 256, args.num_class)
        elif args.cls == 'cos':
            classifier = FC_Classifier(args.feat_dim, args.num_class)
        else:
            classifier = FC_Classifier(args, args.feat_dim)
        classifier.apply(self.weights_init)
        return classifier


class LearningModuleBase(nn.Module):
    def __init__(self):
        super(LearningModuleBase, self).__init__()
        self.range_of_compute = 1

    def forward(self, x):
        raise NotImplementedError

    def _acc(self, pred, label, output='dumb'):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        instance_sum = torch.sum(valid)
        acc = acc_sum.float() / (instance_sum.float() + 1e-10)
        if output == 'dumb':
            del pred
            return acc
        elif output == 'vis':
            return acc, pred, label
        """
        acc_sum = 0
        num = pred.shape[0]
        preds = np.array(pred.detach().cpu())
        preds = np.argsort(preds)
        for i in range(num):
            if label[i] in preds[i, -self.range_of_compute:]:
                acc_sum += 1
        acc = acc_sum / (num + 1e-10)
        return acc
        """


class LearningModule(LearningModuleBase):
    def __init__(self, args, feature_extractor, crit, cls=None, seg=None, output='dumb'):
        super(LearningModule, self).__init__()
        self.feature_extractor = feature_extractor
        self.cls = cls
        self.seg = seg
        self.crit = crit
        self.output = output
        self.sample_per_img = args.sample_per_img

    def forward(self, feed_dict, mode='train', output='dumb'):
        # print(feed_dict['img_data'].shape)
        feature_map = self.feature_extractor(feed_dict['img_data'])
        # print('feature map shape {}'.format(feature_map.shape))
        acc = 0
        loss = 0
        batch_img_num = feature_map.shape[0]

        features = None
        labels = None
        preds = None
        if self.output == 'feat':
            self.cls.output = 'feat'
            for i in range(batch_img_num):
                anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
                if anchor_num == 0 or anchor_num >= 100:
                    continue
                feature = self.cls([feature_map[i], feed_dict['scales'][i], feed_dict['anchors'][i], anchor_num])
                label = feed_dict['cls_label'][i, :anchor_num].long()
                if features is None:
                    features = feature.clone()
                    labels = label.clone()
                else:
                    features = torch.stack((features, feature), dim=0)
                    labels = torch.stack((labels, label), dim=0)
            return features, labels

        if self.output == 'pred':
            self.cls.output = 'pred'
            for i in range(batch_img_num):
                anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
                if anchor_num == 0 or anchor_num >= 100:
                    continue
                pred = self.cls([feature_map[i], feed_dict['scales'][i], feed_dict['anchors'][i], anchor_num]).cpu().data
                label = feed_dict['cls_label'][i, :anchor_num].long().cpu().data
                if preds is None:
                    preds = np.array(pred)
                    labels = np.array(label)
                else:
                    preds = np.vstack((preds, np.array(pred)))
                    labels = np.hstack((labels, np.array(label)))
            return preds, labels

        instance_sum = torch.tensor([0]).cuda()
        for i in range(batch_img_num):
            for crit in self.crit:
                if crit['weight'] == 0:
                    continue
                if self.sample_per_img == -1:  # all the samples are used up
                    anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
                    if anchor_num == 0 or anchor_num >= 100:
                        continue
                    pred = self.cls([feature_map[i], feed_dict['scales'][i], feed_dict['anchors'][i], anchor_num])
                    labels = feed_dict['cls_label'][i, : anchor_num].long()
                    pred = pred.cuda()
                    instance_sum[0] += pred.shape[0]
                    loss += crit['weight'] * crit['crit'](pred, labels) * pred.shape[0]
                    acc += self._acc(pred, labels, self.output) * pred.shape[0]
                    del pred
        return loss / (instance_sum[0] + 1e-10), acc / (instance_sum[0] + 1e-10), instance_sum


class NovelTuningModuleBase(nn.Module):
    def __init__(self):
        super(NovelTuningModuleBase, self).__init__()
        self.range_of_compute = 5

    def forward(self, x):
        raise NotImplementedError

    def _acc(self, pred, label):
        """
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        instance_sum = torch.sum(valid)
        acc = acc_sum.float() / (instance_sum.float() + 1e-10)
        """

        acc_sum = 0
        num = pred.shape[0]
        preds = np.array(pred.detach().cpu())
        preds = np.argsort(preds)
        label = np.array(label.detach().cpu())
        for i in range(num):
            if label[i] in preds[i, -self.range_of_compute:]:
                acc_sum += 1
        acc = torch.tensor(acc_sum / (num + 1e-10)).cuda()
        return acc


class NovelTuningModule(NovelTuningModuleBase):
    def __init__(self, crit, cls=None, seg=None):
        super(NovelTuningModule, self).__init__()
        self.cls = cls
        self.seg = seg
        self.crit = crit
        self.output = 'dumb'

    def forward(self, feed_dict):
        acc = 0
        loss = 0
        for crit in self.crit:
            if crit['weight'] == 0:
                continue
            label = feed_dict['label'].long()
            feature = feed_dict['feature']
            if crit['type'] == 'cls':
                pred = self.cls(feature)
            if self.output != 'dumb':
                return pred
            loss += crit['weight'] * crit['crit'](pred, label)
            acc += self._acc(pred, label)

        return loss, acc
