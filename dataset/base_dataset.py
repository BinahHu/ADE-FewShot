import os
import json
import torch
from dataset.dataset_base import BaseBaseDataset
import cv2
import torchvision
from torchvision import transforms
import numpy as np
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

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
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

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        batch_scales = np.zeros((self.batch_per_gpu, 2), np.float)
        batch_ids = np.zeros(self.batch_per_gpu)
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
            img = cv2.resize(img, (batch_resized_width, batch_resized_height), interp='bilinear')

            # image transform
            img = self.img_transform(img)

            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_ids[i] = this_record[i]['id']

        output = dict()
        output['img_data'] = batch_images
        output['scales'] = batch_scales
        output['ids'] = batch_ids
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass

        
class ObjBaseDataset(BaseBaseDataset):
    """
    Form batch at object level
    """
    def __init__(self, odgt, opt, batch_per_gpu=1, **kwargs):
        super(ObjBaseDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = opt.root_dataset
        self.random_flip = opt.random_flip
        self.mode = opt.sample_type
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu
        self.batch_record_list = []
        # organize objects in categories level
        self.num_class = opt.num_class
        self.cat_list = [[] for i in range(self.num_class)]
        self.construct_cat_list()

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
    """
    def construct_cat_list(self):
        cat_map = {}
        self.cat_sample = []
        for sample in self.list_sample:
            cat = sample['cls_label']
            if cat not in cat_map:
                cat_map[cat] = len(self.cat_sample)
                self.cat_sample.append([])
            self.cat_sample[cat_map[cat]].append(sample)
        self.cat_sample_num = [len(cat) for cat in self.cat_sample]
        self.cat_num = len(self.cat_sample)

    def update_sample_num(self):
        self.cat_sample_num = [len(cat) for cat in self.cat_sample]
    
    def _get_sub_batch_cat(self):
        while True:
            #get a sample record
            cat = self.cur_cat
            this_sample = self.cat_sample[cat][self.cur_cat_idx[cat]]
            self.batch_record_list.append(this_sample)
            
            #update current sample pointer
            self.cur_cat_idx[cat] += 1
            if self.cur_cat_idx[cat] >= self.cat_sample_num[cat]:
                self.cur_cat_idx[cat] = 0
                np.random.shuffle(self.cat_sample[cat])
            
            if len(self.batch_record_list) == self.batch_per_gpu:
                batch_records = self.batch_record_list
                self.batch_record_list = []
                # update current category pointer
                self.cur_cat += 1
                if self.cur_cat >= self.cat_num:
                    self.cur_cat = 0
                    np.random.shuffle(self.cat_sample)
                break
        return batch_records
    """
    def construct_cat_list(self):
        for sample in self.list_sample:
            category = int(sample['cls_label'])
            self.cat_list[category].append(sample)

    def _get_sub_batch_cat(self):
        while True:
            category = random.randint(0, self.num_class - 1)
            index = random.randint(0, len(self.cat_list[category]) - 1)
            self.batch_record_list.append(self.cat_list[category][index])
            if len(self.batch_record_list) == self.batch_per_gpu:
                batch_records = self.batch_record_list
                self.batch_record_list = []
                break
        return batch_records

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
            # np.random.shuffle(self.cat_sample)
            # for cat_list in self.cat_sample:
            #     np.random.shuffle(cat_list)
            # self.update_sample_num()
            self.if_shuffled = True

        # get sub-batch candidates
        if self.mode == 'inst':
            batch_records = self._get_sub_batch()
        elif self.mode == 'cat':
            batch_records = self._get_sub_batch_cat()
        else:
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
