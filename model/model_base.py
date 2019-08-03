import torch
import torch.nn as nn
from model.feature_extractor import LeNet
from model.tail_blocks import FC_Classifier, FC_Classifier2, Cos_Classifier
from model.resnet import resnet18
import math
import numpy as np
import json
import time

attr_table = json.load(open("/home/zhu2/ADE-FewShot/data/ADE/ADE_Origin/attr.json"))

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
            classifier = FC_Classifier(args.feat_dim, 256, args.num_class)
        if args.cls == 'linear2':
            classifier = FC_Classifier2(args.feat_dim, 256, args.num_class)
        elif args.cls == 'cos':
            classifier = Cos_Classifier(args.feat_dim, args.num_class)
        else:
            classifier = FC_Classifier(args.feat_dim, 256, args.num_class)
        classifier.apply(self.weights_init)
        return classifier

    def build_embedding_layer(self, args):
        feat_dim = args.feat_dim
        num_attr = args.num_attr
        if args.is_soft:
            embedder = nn.Embedding(num_attr, feat_dim, padding_idx=0)
        else:
            embedder = nn.Embedding(num_attr+1, feat_dim, padding_idx=0)
        return embedder


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
    def __init__(self, args, feature_extractor, crit, cls=None, embed=None, seg=None, output='dumb'):
        super(LearningModule, self).__init__()
        self.feature_extractor = feature_extractor
        self.cls = cls
        self.seg = seg
        self.crit = crit
        self.output = output
        self.losstype = args.loss
        self.embd = embed
        self.attr_array = []
        self.num_attr = args.num_attr
        #for i in args.gpus:
        #    self.attr_array.append(torch.zeros(args.batch_size_per_gpu, args.num_attr).long().cuda(i))

    def forward(self, feed_dict, mode='train', output='dumb'):
        feature_map = self.feature_extractor(feed_dict['img_data'])
        acc = 0
        loss = 0
        accFlag = False
        for crit in self.crit:
            #Ignore
            if crit['weight'] == 0:
                continue

            #Load Label
            if crit['type'] != 'attr':
                label = feed_dict['{type}_label'.format(type=crit['type'])].long()
            else:
                label = feed_dict['cls_label'].long()



            #cls layer, get pred
            if crit['type'] == 'cls':
                pred = self.cls(feature_map)
            elif crit['type'] == 'attr':
                pred = self.cls(feature_map)



            #loss
            if crit['type'] == 'attr' and self.losstype == 'Attr':
                #attributes = self.attr_array[torch.cuda.current_device()]
                #attributes.zero_()
                attributes = torch.zeros(label.size(0), self.num_attr).long().cuda()
                for i in range(label.size(0)):
                    cat_id = label[i].item()
                    attrs = attr_table[cat_id]
                    for attr in attrs:
                        attributes[i][attr] = 1.0
                embedvec = self.embd(attributes)
                embedvec = embedvec.sum(1).squeeze()
                attr_loss, orth_loss = crit['crit'](feature_map, embedvec)
                loss += crit['weight'] * attr_loss
            else:
                loss_cls = crit['weight'] * crit['crit'](pred, label)
                loss += loss_cls

            #acc
            if accFlag:
                continue
            if self.output == 'dumb':
                acc += self._acc(pred, label, self.output)
            else:
                acc_iter, preds, labels = self._acc(pred, label, self.output)
                acc += acc_iter
            accFlag = True

        if self.output == 'dumb':
            return loss, acc
        elif self.output == 'vis':
            return loss, acc, preds, labels
        elif self.output == 'feat':
            return feature_map, feed_dict['cls_label']


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

            loss += crit['weight'] * crit['crit'](pred, label)
            acc += self._acc(pred, label)

        return loss, acc
