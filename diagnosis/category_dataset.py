import sys
sys.path.append('../')
from dataset.dataset_base import BaseNovelDataset, BaseBaseDataset
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import math
import cv2
import os


class ObjCategoryDataset(BaseBaseDataset):
    def __init__(self, category, odgt, opt, batch_per_gpu=1, **kwargs):
        super(ObjCategoryDataset, self).__init__(odgt, opt, **kwargs)
        self.category = category
        self.root_dataset = opt.root_dataset
        self.category_list = []
        self.batch_per_gpu = batch_per_gpu
        self.batch_record_list = []
        self.parse_into_category()
        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

    def parse_into_category(self):
        for sample in self.list_sample:
            if sample['cls_label'] == self.category:
                self.category_list.append(sample)
        self.num_sample = len(self.category_list)
    
    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            self.batch_record_list.append(this_sample)

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list) == self.batch_per_gpu:
                batch_records = self.batch_record_list
                self.batch_record_list = []
                break

        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        this_short_size = 224
        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        for i in range(self.batch_per_gpu):
            anchor = batch_records[i]['anchor']
            img_height = anchor[1][1] - anchor[0][1]
            img_width = anchor[1][0] - anchor[0][0]
            this_scale = this_short_size / min(img_height, img_width)
            img_resized_height, img_resized_width = \
                math.ceil(img_height * this_scale), math.ceil(img_width * this_scale)
            batch_resized_size[i, :] = img_resized_height, img_resized_width

        batch_images = torch.zeros(self.batch_per_gpu, 3, 224, 224)
        batch_labels = torch.zeros(self.batch_per_gpu).int()
        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]
            anchor = this_record['anchor']

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)[anchor[0][1]:anchor[1][1], anchor[0][0]:anchor[1][0], :]
            assert (img.ndim == 3)

            # note that each sample within a mini batch has different scale param
            img = cv2.resize(img, (batch_resized_size[i, 1], batch_resized_size[i, 0]), interpolation=cv2.INTER_CUBIC)
            # image transform
            img = self.random_crop(img)[0]
            img = self.img_transform(img)

            batch_images[i][:, :, :] = img
            batch_labels[i] = this_record['cls_label']

        output = dict()
        output['img_data'] = batch_images
        output['cls_label'] = batch_labels
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass
