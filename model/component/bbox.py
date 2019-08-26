import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from roi_align.roi_align import RoIAlign
from torch.autograd import Variable


def to_variable(arr, requires_grad=False, is_cuda=True):
    tensor = torch.from_numpy(arr)
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class BBoxModule(nn.Module):
    def __init__(self, args):
        super(BBoxModule, self).__init__()
        self.args = args
        for supervision in args.supervision:
            if supervision['name'] == 'attr':
                self.crop_height = int(supervision['other']['pool_size'])
                self.crop_width = int(supervision['other']['pool_size'])
        self.roi_align = RoIAlign(self.crop_height, self.crop_width, transform_fpcoor=True)

        self.feat_dim = args.feat_dim * self.crop_width * self.crop_height
        self.num_class = args.num_base_class
        self.regress = nn.Linear(self.feat_dim, 4)

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

    @staticmethod
    def compute_anchor_location(anchor, scale, original_scale):
        """
        compute the anchor location after resize operation
        :param anchor: input anchor
        :param scale: scale
        :param original_scale: the scale of original data loading
        :return: anchor on the feature map
        """
        anchor = np.array(anchor.detach().cpu())
        original_scale = np.array(original_scale)
        scale = np.array(scale.cpu())
        anchor[:, 2] = anchor[:, 2] * scale[0] * original_scale[0]
        anchor[:, 3] = anchor[:, 3] * scale[0] * original_scale[0]
        anchor[:, 0] = anchor[:, 0] * scale[1] * original_scale[1]
        anchor[:, 1] = anchor[:, 1] * scale[1] * original_scale[1]
        return anchor

    @staticmethod
    def prepare_target_value(crop_anchor, tgt_anchor):
        crop_anchor = np.array(crop_anchor.cpu())
        tgt_anchor = np.array(tgt_anchor.cpu())
        # transpose the anchors, then we can directly get the location on each axis
        crop_anchor = np.transpose(crop_anchor)
        tgt_anchor = np.transpose(tgt_anchor)
        # left, right, up, down
        x_crop, r_crop, y_crop, d_crop = crop_anchor
        x_tgt, r_tgt, y_tgt, d_tgt = tgt_anchor
        h_crop = d_crop - y_crop
        h_tgt = d_tgt - y_tgt
        w_crop = r_crop - x_crop
        w_tgt = r_tgt - x_tgt

        # compute the value
        dx = (x_tgt - x_crop) / w_crop
        dy = (y_crop - y_tgt) / y_crop
        dw = np.log(w_tgt / w_crop)
        dh = np.log(h_tgt / h_crop)
        target_value = np.stack((dx, dy, dw, dh), axis=0)
        # restore the shape
        target_value = np.transpose(target_value)
        return target_value

    def forward(self, agg_data):
        feature_map = agg_data['feature_map']
        crop_anchors = agg_data['anchors']
        anchor_num = crop_anchors.shape[0]
        tgt_anchors = agg_data['bbox'][:anchor_num, :]
        scale = agg_data['scales']
        features = self.process_in_roi_layer(feature_map, scale,
                                            crop_anchors, anchor_num)

        target_value = self.prepare_target_value(crop_anchor=crop_anchors, tgt_anchor=tgt_anchors)
        target_value = torch.tensor(target_value).cuda()

        pred_value = self.regress(features)
        loss = F.smooth_l1_loss(pred_value, target_value)
        return loss




