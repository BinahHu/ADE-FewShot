"""
Dataset for novel classes
"""
import os
import json
import torch
from dataset.proto_dataset import NovelProtoDataset
import cv2
import math
import numpy as np
import h5py
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='/home/zpang/ADE-FewShot/log.txt',
                    filemode='a')


class NovelDataset(NovelProtoDataset):
    """
    Form batch at object level
    """
    def __init__(self, h5path, opt, batch_per_gpu=1, **kwargs):
        super(NovelDataset, self).__init__(h5path, opt, **kwargs)
        self.batch_per_gpu = batch_per_gpu
        self.batch_record_list = []

        # override dataset length when training with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

        self.crop_height = opt.crop_height
        self.crop_width = opt.crop_width
        self.feat_dim = opt.feat_dim

    def _get_sub_batch(self):
        while True:
            if self.args.base_shot == 10:
                category = np.random.randint(0, self.args.num_novel_class)
                instance_location = np.random.randint(self.start_location[category],
                                                      self.start_location[category] + self.category_num[category])
                # print(instance_location)
                this_sample = dict()
                this_sample['label'] = self.labels[instance_location].copy()
                file_index = instance_location // 100
                f = h5py.File('data_file/{}_{}/{}_feat_{}.h5'.format(self.args.model, self.args.base_shot,
                                                                     self.part, file_index), 'r')
                feature = np.array(f['feature_map'][instance_location % 100, :])
                this_sample['feature'] = feature.copy()
                self.batch_record_list.append(this_sample)
                logging.debug('{},{}'.format(this_sample['label'], instance_location))
                if len(self.batch_record_list) == self.batch_per_gpu:
                    batch_records = self.batch_record_list
                    self.batch_record_list = []
                    break
            else:
                instance_location = self.index_list[self.cur_idx]
                this_sample = dict()
                this_sample['label'] = self.labels[instance_location].copy()
                file_index = instance_location // 100
                f = h5py.File('data_file/{}_{}/{}_feat_{}.h5'.format(self.args.model, self.args.base_shot,
                                                                     self.part, file_index), 'r')
                feature = np.array(f['feature_map'][instance_location % 100, :])
                this_sample['feature'] = feature.copy()
                self.batch_record_list.append(this_sample)

                # update current sample pointer
                self.cur_idx += 1
                if self.cur_idx >= self.num_sample:
                    self.cur_idx = 0
                    np.random.shuffle(self.index_list)

                if len(self.batch_record_list) == self.batch_per_gpu:
                    batch_records = self.batch_record_list
                    self.batch_record_list = []
                    break

        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.index_list)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_features = torch.zeros(self.batch_per_gpu, self.feat_dim * self.crop_width * self.crop_height)
        batch_labels = torch.zeros(self.batch_per_gpu).int()
        for i in range(self.batch_per_gpu):
            batch_features[i] = torch.tensor(batch_records[i]['feature'])
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
