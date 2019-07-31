"""
Dataset for novel classes
"""
import os
import json
import torch
from dataset.dataset_base import BaseNovelDataset
import cv2
import math
import numpy as np

class ObjNovelDataset(BaseNovelDataset):
    """
    Form batch at object level
    """

    def __init__(self, h5path, opt, batch_per_gpu=1, **kwargs):
        super(ObjNovelDataset, self).__init__(h5path, opt, **kwargs)
        self.batch_per_gpu = batch_per_gpu
        self.batch_record_list = []

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

        self.crop_height = opt.crop_height
        self.crop_width = opt.crop_width

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.data[self.cur_idx]
            self.batch_record_list.append(this_sample)

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.data)

            if len(self.batch_record_list) == self.batch_per_gpu:
                batch_records = self.batch_record_list
                self.batch_record_list = []
                break

        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.data)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        this_short_size = 224

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_features = torch.zeros(self.batch_per_gpu, self.feat_dim * self.crop_width * self.crop_height)
        batch_labels = torch.zeros(self.batch_per_gpu).int()
        batch_anchors = torch.zeros(self.batch_per_gpu, 4)
        batch_scales = torch.zeros(self.batch_per_gpu, 2)
        for i in range(self.batch_per_gpu):
            batch_features[i] = torch.tensor(batch_records[i]['feature']).view(-1)
            batch_labels[i] = torch.tensor(batch_records[i]['label'].astype(np.float)).int()
            # batch_anchors[i] = torch.tensor(batch_records[i]['anchors'])
            # batch_scales[i] = torch.tensor(batch_records[i]['scales'])

        output = dict()
        output['feature'] = batch_features
        output['label'] = batch_labels
        # output['anchors'] = batch_anchors
        # output['scales'] = batch_scales
        return output

    def __len__(self):
        return int(1e10)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass
