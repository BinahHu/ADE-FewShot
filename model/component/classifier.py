import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='train.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s : %(message)s'
                    #日志格式
                    )

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.num_class = args.num_base_class
        self.down_sampling_rate = args.down_sampling_rate
        self.fc = nn.Linear(args.feat_dim * args.crop_width * args.crop_height, self.num_class)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mode = 'train'

    def _acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        if preds.get_device() == 2:
            logging.info("Preds " + str(preds.tolist()))
            logging.info("Labels " + str(label.tolist()))
        acc_sum = torch.sum(valid * (preds == label).long())
        instance_sum = torch.sum(valid)
        acc = acc_sum.float() / (instance_sum.float() + 1e-10)
        del pred
        return acc

    def diagnosis(self, x):
        """
        used only when diagnosis
        :param x: input
        :return: prediction
        """
        feature, labels = x
        pred = self.fc(feature)
        return pred

    def forward(self, x):
        if self.mode == 'diagnosis':
            return self.diagnosis(x)

        feature, labels = x
        pred = self.fc(feature)
        loss = self.loss(pred, labels)
        acc = self._acc(pred, labels)
        return loss, acc
