"""
Transform data in supervision from raw form into usable data
"""
import numpy as np
import cv2
import torch
import os
import json


class Transform:
    def __init__(self, args):
        self.args = args

    def seg_transform(self, path, other=None):
        """
        segmentation transform
        :param path: segmentation map path
        :return: segmentation map in the original size
        """
        path = os.path.join(self.args.root_dataset, path)
        img = cv2.imread(path, 0)
        return img

    def attr_transform(self, tensor, other=None):
        """
        attribute transform
        :param tensor: input attribute list
        :param other: other information needed for transformation
        :return: hot result
        """
        if other is None:
            raise Exception('No attribute num for attribute supervision')
        attr_num = other['num_attr']
        result = np.zeros(attr_num).astype(np.int)
        for i in tensor:
            result[i] = 1
        return result