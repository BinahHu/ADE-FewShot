import torch
import torch.nn as nn
import numpy as np
from roi_align.roi_align import RoIAlign
from torch.autograd import Variable
import random
from .component.resnet import Bottleneck

def to_variable(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

def build_task_head(in_channels, out_channels):
    downsample = None
    if in_channels != out_channels:
        downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                             stride=1, bias=False), nn.BatchNorm2d(out_channels))

    bottleneck1 = Bottleneck(in_channels, out_channels // 4, downsample=downsample)
    return bottleneck1
    #bottleneck2 = Bottleneck(out_channels, out_channels // 4, downsample=None)
    #return nn.Sequential(bottleneck1, bottleneck2)

class BaseLearningModule(nn.Module):
    def __init__(self, args, backbone, initial_classifier, classifier, distillation):
        super(BaseLearningModule, self).__init__()
        self.args = args
        self.backbone = backbone
        self.initial_classifier = initial_classifier
        self.classifier = classifier
        self.distillation = distillation
        self.enable_distillation = args.distillation

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

        task_specific_head = {}
        task_specific_head['classify'] = build_task_head(args.feat_dim, args.feat_dim)
        for supervision in args.supervision:
            name = supervision['name']
            task_specific_head[name] = build_task_head(args.feat_dim, args.feat_dim)

        self.task_specific_head = nn.ModuleDict(task_specific_head)

        self.mode = 'train'
        if self.initial_classifier is not None:
            self.initial_classifier.mode = self.mode

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
        feature_map = self.task_specific_head['classify'](feature_map)
        feature_map_dict = {}
        feature_map_dict['classify'] = self.task_specific_head['classify'](feature_map)
        for supervision in self.args.supervision:
            name = supervision['name']
            feature_map_dict[name] = self.task_specific_head[name](feature_map)

        if self.enable_distillation:
            feature_map = self.distillation(feature_map_dict)

        batch_img_num = feature_map.shape[0]
        features = None
        labels = None
        for i in range(batch_img_num):
            anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
            if anchor_num == 0 or anchor_num > 100:
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
        if features.shape[0] != labels.shape[0]:
            print(features.shape[0])
            print(anchor_num)
        return features, labels

    def forward(self, feed_dict):
        if self.mode == 'feature':
            return self.predict(feed_dict)
        elif self.mode == 'diagnosis':
            return self.diagnosis(feed_dict)



        feature_map = self.backbone(feed_dict['img_data'])
        feature_map_dict = {}
        feature_map_dict['classify'] = self.task_specific_head['classify'](feature_map)
        for supervision in self.args.supervision:
            name = supervision['name']
            feature_map_dict[name] = self.task_specific_head[name](feature_map)

        if self.enable_distillation:
            feature_map_dict['final'] = self.distillation(feature_map_dict)

        loss = 0
        batch_img_num = feature_map.shape[0]
        instance_sum = torch.tensor([0]).cuda()

        acc_initial = 0
        category_accuracy_initial = torch.zeros(2, self.args.num_base_class).cuda()
        loss_classification_initial = torch.zeros(1)

        acc_final = 0
        category_accuracy_final = torch.zeros(2, self.args.num_base_class).cuda()
        loss_classification_final = torch.zeros(1)

        loss_supervision = torch.zeros(len(self.args.supervision))
        for i in range(batch_img_num):
            anchor_num = int(feed_dict['anchor_num'][i].detach().cpu())
            if anchor_num == 0 or anchor_num > 100:
                continue
            
            feature_dict = {}
            feature_dict['classify'] = self.process_in_roi_layer(feature_map_dict['classify'][i], feed_dict['scales'][i],
                                                feed_dict['anchors'][i], anchor_num)
            for supervision in self.args.supervision:
                name = supervision['name']
                feature_dict[name] = self.process_in_roi_layer(feature_map_dict[name][i], feed_dict['scales'][i],
                                                feed_dict['anchors'][i], anchor_num)

            if self.enable_distillation:
                feature_dict['final'] = self.process_in_roi_layer(feature_map_dict['final'][i],
                                                          feed_dict['scales'][i], feed_dict['anchors'][i], anchor_num)
            
            labels = feed_dict['label'][i, : anchor_num].long()
            loss_cls_initial, acc_cls_initial, category_acc_img_initial = self.initial_classifier([feature_dict['classify'], labels])
            instance_sum[0] += labels.shape[0]
            loss += loss_cls_initial * labels.shape[0]

            acc_initial += acc_cls_initial * labels.shape[0]
            loss_classification_initial += loss_cls_initial.item() * labels.shape[0]
            category_accuracy_initial += category_acc_img_initial.cuda()

            if self.enable_distillation:
                loss_cls_final, acc_cls_final, category_acc_img_final = self.classifier(
                    [feature_dict['final'], labels])
                loss += loss_cls_final * labels.shape[0]
                acc_final += acc_cls_final * labels.shape[0]
                loss_classification_final += loss_cls_final.item() * labels.shape[0]
                category_accuracy_final += category_acc_img_final.cuda()


            # do not contain other supervision
            if not hasattr(self.args, 'module'):
                continue
            if self.mode == 'val':
                continue

            # form generic data input for all supervision branch
            input_agg = dict()

            input_agg['anchors'] = feed_dict['anchors'][i][:anchor_num]
            input_agg['scales'] = feed_dict['scales'][i]
            input_agg['labels'] = feed_dict['label'][i][:anchor_num]

            for key in feed_dict.keys():
                if key not in ['img_data', 'patch_location_label', 'patch_location__img',
                               'rotation_img', 'rotation_label']:
                    supervision = next((x for x in self.args.supervision if x['name'] == key), None)
                    if supervision is not None:
                        input_agg[key] = feed_dict[key][i]

            for j, supervision in enumerate(self.args.supervision):
                name = supervision['name']
                input_agg['features'] = feature_dict[name]
                input_agg['feature_map'] = feature_map_dict[name][i]
                if supervision['type'] != 'self':
                    loss_branch = getattr(self, supervision['name'])(input_agg) * labels.shape[0]
                elif supervision['name'] == 'patch_location':
                    input_patch_location = feed_dict['patch_location_img']
                    _, _, _, height, width = input_patch_location.shape
                    patch_location_label = feed_dict['patch_location_label']
                    patch_location_feature_map = self.backbone(input_patch_location.view(-1, 3, height, width))
                    _, C, H, W = patch_location_feature_map.shape
                    patch_location_feature_map = patch_location_feature_map.reshape(batch_img_num, 2, C, H, W)
                    loss_branch = getattr(self, 'patch_location')([patch_location_feature_map, patch_location_label])
                elif supervision['name'] == 'rotation':
                    input_img = feed_dict['rotation_img']
                    input_label = feed_dict['rotation_label']
                    rotation_feature_map = self.backbone(input_img)
                    loss_branch = getattr(self, 'rotation')([rotation_feature_map, input_label])
                loss += (loss_branch * supervision['weight'])
                loss_supervision[j] += loss_branch.item()

        if self.mode == 'val':
            return category_accuracy_initial, category_accuracy_final, loss / (instance_sum[0] + 1e-10), \
                   acc_initial / (instance_sum[0] + 1e-10), acc_final / (instance_sum[0] + 1e-10), instance_sum
        if hasattr(self.args, 'module'):
            loss_supervision = loss_supervision.cuda()
            loss_classification_initial = loss_classification_initial.cuda()
            loss_classification_final = loss_classification_final.cuda()
            return category_accuracy_initial, category_accuracy_final, loss / (instance_sum[0] + 1e-10), \
                   acc_initial / (instance_sum[0] + 1e-10), acc_final / (instance_sum[0] + 1e-10), instance_sum, \
                   loss_supervision / (instance_sum[0] + 1e-10), \
                   loss_classification_initial / (instance_sum[0] + 1e-10), \
                   loss_classification_final / (instance_sum[0] + 1e-10)
        else:
            return category_accuracy_initial, category_accuracy_final, loss / (instance_sum[0] + 1e-10), \
                   acc_initial / (instance_sum[0] + 1e-10), acc_final / (instance_sum[0] + 1e-10), \
                   instance_sum, None, None, None