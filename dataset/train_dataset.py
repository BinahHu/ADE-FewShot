"""
Classes for the datasets in ADE20K
Modified from https://github.com/CSAILVision/semantic-segmentation-pytorch
"""
import os
import torch
import json
import cv2
from torchvision import transforms
import numpy as np
import dataset.dataset_base as dataset_base


class TrainDataset(dataset_base.BaseDataset):
    """
    Base class for the training of feature extractor
    """
    def __init__(self, odgt, opt, batch_per_gpu=1, **kwargs):
        """
        :param odgt: list of items
        :param opt: parameters
        :param batch_per_gpu: batch size (see train.py)
        :param kwargs: other arguments
        """
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = opt.root_dataset
        self.random_shuffle = opt.random_shuffle
        self.batch_per_gpu = batch_per_gpu
        self.cur_idx = 0
        self.if_shuffled = False

    def _get_sub_batch(self):
        return NotImplementedError

    def __getitem__(self, index):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError



