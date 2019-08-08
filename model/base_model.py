import torch
import torch.nn as nn
import numpy as np
from roi_align.roi_align import RoIAlign
from torch.autograd import Variable
import logging
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='train.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(levelname)s: %(message)s'
                    #日志格式
                    )



def to_variable(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class BaseLearningModule(nn.Module):
    def __init__(self, args, backbone, classifier):
        super(BaseLearningModule, self).__init__()
        self.args = args
        self.backbone = backbone
        self.classifier = classifier

        self.crop_height = int(args.crop_height)
        self.crop_width = args.crop_width
        self.roi_align = RoIAlign(args.crop_height, args.crop_width, transform_fpcoor=True)
        self.down_sampling_rate = args.down_sampling_rate

        # supervision modules are generated in train
        # args.module example:
        # [{'name': 'seg', 'module': seg_module}]
        if hasattr(args, 'module'):
            for module in args.module:
                setattr(self, module['name'], module['module'])

        self.mode = 'train'
        if self.classifier is not None:
            self.classifier.mode = self.mode

    def process_in_roi_layer(self, feature_map, scale, anchors, anchor_num):
        """
        process the data in roi_layer and get the feature
        :param feature_map: C * H * W
        :param scale: anchor_num * 2
        :param anchors: anchor_num * 4
        :param anchor_num: int
        :return: feature C * crop_height * crop_width
        """
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
        feature_map = feature_map.unsqueeze(0)
        feature = self.roi_align(feature_map, anchors, anchor_index)
        feature = feature.view(-1, self.args.feat_dim * self.crop_height * self.crop_width)
        return feature

    def predict(self, feed_dict):
        feature_map = self.backbone(feed_dict['img_data'])
        batch_img_num = feature_map.shape[0]
        features = None
        labels = None
        for i in range(batch_img_num):
            anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
            if anchor_num == 0 or anchor_num >= 100:
                continue
            feature = self.process_in_roi_layer(feature_map[i], feed_dict['scales'][i],
                                                feed_dict['anchors'][i], anchor_num)
            label = feed_dict['label'][i][:anchor_num].long()

            if features is None:
                features = feature.clone()
                labels = label.clone()
            else:
                features = torch.stack((features, feature), dim=0)
                labels = torch.stack((labels, label), dim=0)
        return features, labels

    def diagnosis(self, feed_dict):
        """
        used only when computing
        :param feed_dict:
        :return: prediction and labels
        """
        self.classifier.mode = 'diagnosis'
        feature_map = self.backbone(feed_dict['img_data'])
        batch_img_num = feature_map.shape[0]
        predictions = None
        labels = None
        for i in range(batch_img_num):
            anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
            if anchor_num == 0 or anchor_num >= 100:
                continue
            feature = self.process_in_roi_layer(feature_map[i], feed_dict['scales'][i],
                                                feed_dict['anchors'][i], anchor_num)
            label = feed_dict['label'][i][:anchor_num].long()
            prediction = self.classifier([feature, label])

            if predictions is None:
                predictions = prediction.clone()
                labels = label.clone()
            else:
                predictions = torch.stack((predictions, prediction), dim=0)
                labels = torch.stack((labels, label), dim=0)
        return predictions, labels

    def forward(self, feed_dict):
        if self.mode == 'feature':
            return self.predict(feed_dict)
        elif self.mode == 'diagnosis':
            return self.diagnosis(feed_dict)

        feature_map = self.backbone(feed_dict['img_data'])
        acc = 0
        loss = 0
        batch_img_num = feature_map.shape[0]

        instance_sum = torch.tensor([0]).cuda()
        loss_classification = torch.zeros(1)
        loss_supervision = torch.zeros(len(self.args.supervision))
        for i in range(batch_img_num):
            anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
            if anchor_num == 0 or anchor_num >= 100:
                continue
            feature = self.process_in_roi_layer(feature_map[i], feed_dict['scales'][i],
                                                feed_dict['anchors'][i], anchor_num)
            labels = feed_dict['label'][i, : anchor_num].long()
            if labels.get_device() == 999:
                logging.info("Model " + str(labels))
            loss_cls, acc_cls = self.classifier([feature, labels])
            instance_sum[0] += labels.shape[0]
            loss += loss_cls * labels.shape[0]
            acc += acc_cls * labels.shape[0]
            loss_classification += loss_cls.item() * labels.shape[0]
            # do not contain other supervision
            if not hasattr(self.args, 'module'):
                continue
            if self.mode == 'val':
                continue

            # form generic data input for all supervision branch
            input_agg = dict()
            input_agg['features'] = feature
            input_agg['feature_map'] = feature_map[i]
            for key in feed_dict.keys():
                if key not in ['img_data']:
                    supervision = next((x for x in self.args.supervision if x['name'] == key), None)
                    if (supervision is not None) and (supervision['type'] == 'inst'):
                        input_agg[key] = feed_dict[key][i]

            # process through each branch
            for j, supervision in enumerate(self.args.supervision):
                loss_branch = getattr(self, supervision['name'])(input_agg) * labels.shape[0]
                loss += (loss_branch * supervision['weight'])
                loss_supervision[j] += loss_branch.item()

        if self.mode == 'val':
            return loss / (instance_sum[0] + 1e-10), acc / (instance_sum[0] + 1e-10), instance_sum
        if hasattr(self.args, 'module'):
            loss_supervision = loss_supervision.cuda()
            loss_classification = loss_classification.cuda()
            return loss / (instance_sum[0] + 1e-10), acc / (instance_sum[0] + 1e-10), instance_sum, \
                   loss_supervision / (instance_sum[0] + 1e-10), loss_classification / (instance_sum[0] + 1e-10)
        else:
            return loss / (instance_sum[0] + 1e-10), acc / (instance_sum[0] + 1e-10), instance_sum, None, None
