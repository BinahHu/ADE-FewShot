import os
import json
import torch
from dataset.dataset_base import BaseTrainDataset
import cv2
import torchvision
from torchvision import transforms
import numpy as np
import math


class ImgTrainDataset(BaseTrainDataset):
    """
    Form a batch with original images
    """
    def __init__(self, odgt, opt, batch_per_gpu=1, **kwargs):
        super(ImgTrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = opt.root_dataset
        self.random_flip = opt.random_flip
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into classes according to their ratio between height and width
        self.group_split = (opt.group_split).copy()
        self.worst_ratio = opt.worst_ratio
        self.group_split.append(self.worst_ratio)
        self.group_split.insert(0, 1 / self.worst_ratio)
        self.batch_record_list = [[] for i in range(len(self.group_split) - 1)]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
        self.crop = opt.crop

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            height = this_sample['height']
            width = this_sample['width']
            for i in range(len(self.group_split) - 1):
                if self.group_split[i] < width / height <= self.group_split[i + 1]:
                    self.batch_record_list[i].append(this_sample)

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            # if there are enough batch candidates already
            flag = 0
            for i, record_list in enumerate(self.batch_record_list):
                if len(record_list) == self.batch_per_gpu:
                    batch_records = self.batch_record_list[i]
                    self.batch_record_list[i] = []
                    flag = 1
                    break
            if flag == 1:
                break

        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        if self.crop == False:
            # resize all images' short edges to the chosen size
            if isinstance(self.imgSize, list):
                if len(self.imgSize >= 1):
                    this_short_size = np.random.choice(self.imgSize)
            else:
                this_short_size = self.imgSize

            # calculate the BATCH's height and width
            # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
            batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
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

            assert self.padding_constant >= self.segm_downsampling_rate, \
                'padding constant must be equal or large than segm downsamping rate'
            batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)
            batch_segms = torch.zeros(self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate,
                                      batch_resized_width // self.segm_downsampling_rate).long()

            for i in range(self.batch_per_gpu):
                this_record = batch_records[i]
                height = this_record['height']
                width = this_record['width']

                # load image and label
                image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
                segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)[:height, :width, :]
                segm = cv2.imread(segm_path, cv2.IMREAD_COLOR)[:height, :width, :]

                assert(img.ndim == 3)
                assert(segm.ndim == 3)
                assert(img.shape[0] == segm.shape[0])
                assert(img.shape[1] == segm.shape[1])

                if self.random_flip is True:
                    random_flip = np.random.choice([0, 1])
                    if random_flip == 1:
                        img = cv2.flip(img, 1)
                        segm = cv2.flip(segm, 1)

                # note that each sample within a mini batch has different scale param
                img = cv2.resize(img, (batch_resized_size[i, 1], batch_resized_size[i, 0]),
                                 interpolation=cv2.INTER_LINEAR)
                segm = cv2.resize(segm, (batch_resized_size[i, 1], batch_resized_size[i, 0]),
                                  interpolation=cv2.INTER_NEAREST)

                # to avoid seg label misalignment
                segm_rounded_height = self.round2nearest_multiple(segm.shape[0], self.segm_downsampling_rate)
                segm_rounded_width = self.round2nearest_multiple(segm.shape[1], self.segm_downsampling_rate)
                segm_rounded =     np.zeros((segm_rounded_height, segm_rounded_width, 3), dtype='uint8')
                segm_rounded[:segm.shape[0], :segm.shape[1], :] = segm

                segm = cv2.resize(
                    segm_rounded,
                    (segm_rounded.shape[1] // self.segm_downsampling_rate,
                     segm_rounded.shape[0] // self.segm_downsampling_rate),
                    interpolation=cv2.INTER_NEAREST)
                segm[:, :, 0] = segm[:, :, 1].astype(np.int) + ((segm[:, :, 2] / 10) * 256).astype(np.int)
                segm = segm[:, :, 0]

                # image transform
                img = self.img_transform(img)

                batch_images[i][:, :img.shape[1], :img.shape[2]] = img
                batch_segms[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        batch_segms = batch_segms - 1 # label from -1 to 149
        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ObjTrainDataset(BaseTrainDataset):
    """
    Form batch at object level
    """
    def __init__(self, odgt, opt, batch_per_gpu=1, **kwargs):
        super(ObjTrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = opt.root_dataset
        self.random_flip = opt.random_flip
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into classes according to their ratio between height and width
        self.group_split = (opt.group_split).copy()
        self.worst_ratio = opt.worst_ratio
        self.group_split.append(self.worst_ratio)
        self.group_split.insert(0, 1 / self.worst_ratio)
        self.batch_record_list = [[] for i in range(len(self.group_split) - 1)]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False
        self.crop = opt.crop

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            anchor = this_sample['anchor']
            height = anchor[1][1] - anchor[0][1]
            width = anchor[1][0] - anchor[0][0]
            for i in range(len(self.group_split) - 1):
                if self.group_split[i] < width / height <= self.group_split[i + 1]:
                    self.batch_record_list[i].append(this_sample)

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            flag = 0
            for i, record_list in enumerate(self.batch_record_list):
                if len(record_list) == self.batch_per_gpu:
                    batch_records = self.batch_record_list[i]
                    self.batch_record_list[i] = []
                    flag = 1
                    break
            if flag == 1:
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
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize

        if self.crop:
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
            assert(img.ndim == 3)

            if self.random_flip is True:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)

            # note that each sample within a mini batch has different scale param
            img = cv2.resize(img, (batch_resized_size[i, 1], batch_resized_size[i, 0]), interpolation=cv2.INTER_LINEAR)
            img = self.random_crop(img)
            # image transform
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
