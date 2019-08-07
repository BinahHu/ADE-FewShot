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

class ObjBaseDataset(BaseBaseDataset):
    """
    Form batch at object level
    """
    def __init__(self, odgt, opt, batch_per_gpu=1, **kwargs):
        super(ObjBaseDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = opt.root_dataset
        self.random_flip = opt.random_flip
        self.mode = opt.sample_type
        self.loss = opt.loss
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu
        self.batch_record_list = []
        # organize objects in categories level
        self.num_class = opt.num_class
        if self.mode is not 'inst':
            self.cat_list = [[] for i in range(self.num_class)]
            self.cat_length = np.zeros(self.num_class)
            self.cat_weight = np.zeros(self.num_class)
            self.construct_cat_list(opt)

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
        self.attr_num = opt.num_attr

    def construct_cat_list(self, args):
        def weight_function(x, args):
            if args.sample_type == 'cat_sqrt':
                return math.sqrt(x)
            elif args.sample_type == 'cat_equal':
                return 1
            elif args.sample_type == 'inst':
                return x
        for sample in self.list_sample:
            category = int(sample['cls_label'])
            self.cat_list[category].append(sample)

        for i in range(self.num_class):
            self.cat_length[i] = len(self.cat_list[i])
            self.cat_weight[i] = weight_function(self.cat_length[i], args)
        weight_sum = np.sum(self.cat_weight)
        for i in range(self.num_class):
            self.cat_weight[i] = self.cat_weight[i] / weight_sum

    def _get_sub_batch_cat(self):
        batch_records = []
        sample_categories = choice(np.arange(self.num_class).astype(np.int),
                                   self.batch_per_gpu,
                                   p=self.cat_weight,
                                   replace=False)
        for sample_category in sample_categories:
            length = len(self.cat_list[sample_category])
            batch_records.append(self.cat_list[sample_category][random.randint(0, length - 1)])
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
        if self.loss == 'Attr':
            batch_attrs = torch.zeros(self.batch_per_gpu, self.attr_num).int()
        if self.loss == 'Multi':
            batch_labels = -1 * torch.ones(self.batch_per_gpu, self.num_class).int()
        else:
            batch_labels = torch.zeros(self.batch_per_gpu).int()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]
            anchor = this_record['anchor']
            if self.loss == 'Attr':
                attr_record = this_record['attr']
                for attr in attr_record:
                    batch_attrs[i][attr] = attr + 1

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)[anchor[0][1]:anchor[1][1], anchor[0][0]:anchor[1][0], :]
            assert (img.ndim == 3)
            # note that each sample within a mini batch has different scale param
            # img = cv2.resize(img, (batch_resized_size[i, 1], batch_resized_size[i, 0]), interpolation=cv2.INTER_CUBIC)
            if self.mode == 'val':
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            elif self.mode == 'train':
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            img = self.img_transform(img)

            batch_images[i][:, :, :] = img
            if self.loss != 'Multi':
                batch_labels[i] = this_record['cls_label']
            else:
                labels = torch.tensor(this_record['multi_label'])
                batch_labels[i, :labels.size] = labels

        output = dict()
        output['img_data'] = batch_images
        output['cls_label'] = batch_labels
        if self.loss == 'Attr':
            output['attr'] = batch_attrs
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass
