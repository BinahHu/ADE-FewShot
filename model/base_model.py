import torch
import torch.nn as nn
import numpy as np


class BaseLearningModule(nn.Module):
    def __init__(self, args, backbone, classifier):
        super(BaseLearningModule, self).__init__()
        self.args = args
        self.backbone = backbone
        self.classifier = classifier
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.output = 'dumb'

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

    def forward(self, feed_dict):
        feature_map = self.backbone(feed_dict['img_data'])
        acc = 0
        loss = 0
        batch_img_num = feature_map.shape[0]

        features = None
        labels = None
        if self.output == 'feat':
            self.cls.output = 'feat'
            for i in range(batch_img_num):
                anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
                if anchor_num == 0 or anchor_num >= 100:
                    continue
                feature = self.classifier([feature_map[i], feed_dict['scales'][i], feed_dict['anchors'][i], anchor_num])
                label = feed_dict['cls_label'][i, :anchor_num].long()
                if features is None:
                    features = feature.clone()
                    labels = label.clone()
                else:
                    features = torch.stack((features, feature), dim=0)
                    labels = torch.stack((labels, label), dim=0)
            return features, labels

        instance_sum = torch.tensor([0]).cuda()
        for i in range(batch_img_num):
            anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
            if anchor_num == 0 or anchor_num >= 100:
                continue
            pred, feature = self.classifier([feature_map[i], feed_dict['scales'][i], feed_dict['anchors'][i], anchor_num])
            labels = feed_dict['label'][i, : anchor_num].long()
            instance_sum[0] += pred.shape[0]
            pred = pred.cuda()
            loss += self.loss(pred, labels) * pred.shape[0]
            acc += self._acc(pred, labels, self.output) * pred.shape[0]
        return loss / (instance_sum[0] + 1e-10), acc / (instance_sum[0] + 1e-10), instance_sum
