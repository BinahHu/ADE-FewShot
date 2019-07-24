import os
import json
import torch
from dataset.dataset_base import BaseBaseDataset
import cv2
import torchvision
from torchvision import transforms
import numpy as np
from numpy.random import choice
import math
import random


class ImgBaseDataset(BaseBaseDataset):
    """
    Form batch at object level
    """
    def __init__(self, odgt, opt, batch_per_gpu=1, **kwargs):
        super(ImgBaseDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = opt.root_dataset
        self.random_flip = opt.random_flip
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
        self.max_anchor_per_img = opt.max_anchor_per_img

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            # print(len(this_sample['anchors']))
            if len(this_sample['anchors']) == 0:
                self.cur_idx += 1
                if self.cur_idx >= self.num_sample:
                    self.cur_idx = 0
                    np.random.shuffle(self.list_sample)
                continue
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample)  # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample)  # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        this_short_size = 500

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        batch_scales = np.zeros((self.batch_per_gpu, 2), np.float)
        batch_anchor_num = np.zeros(self.batch_per_gpu)
        batch_labels = np.zeros((self.batch_per_gpu, self.max_anchor_per_img))
        batch_anchors = np.zeros((self.batch_per_gpu, self.max_anchor_per_img, 4))
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(self.round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(self.round2nearest_multiple(batch_resized_width, self.padding_constant))

        for i in range(self.batch_per_gpu):
            batch_scales[i, 0] = batch_resized_height / batch_records[i]['height']
            batch_scales[i, 1] = batch_resized_width / batch_records[i]['width']

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            assert (img.ndim == 3)
            # note that each sample within a mini batch has different scale param
            img = cv2.resize(img, (batch_resized_width, batch_resized_height), interpolation=cv2.INTER_CUBIC)
            # image transform
            img = self.img_transform(img)
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            anchors = batch_records[i]['anchors']
            anchor_num = min(len(anchors), self.max_anchor_per_img)
            batch_anchor_num[i] = anchor_num
            for j in range(anchor_num):
                batch_labels[i, j] = int(anchors[j]['cls_label'])
                batch_anchors[i, j, :] = np.array(anchors[j]['anchor'])

        output = dict()
        output['img_data'] = batch_images
        output['scales'] = torch.tensor(batch_scales)
        output['cls_label'] = torch.tensor(batch_labels)
        output['anchors'] = torch.tensor(batch_anchors)
        output['anchor_num'] = torch.tensor(batch_anchor_num)

        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass
