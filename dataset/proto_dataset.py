import bisect
import warnings
from torch._utils import _accumulate
from torch import randperm
import torch
from torchvision import transforms
import numpy as np
import json
import random
import h5py
import cv2


class Dataset(object):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class TensorDataset(Dataset):
    """Dataset wrapping data and target tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (iterable): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths
    ds
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (iterable): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class BaseProtoDataset(Dataset):
    def __init__(self, data_file, args):
        # parse options
        self.imgShortSize = args.imgShortSize
        self.imgMaxSize = args.imgMaxSize
        self.list_sample = None
        self.num_sample = 0
        self.args = args

        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = args.padding_constant
        # parse the input list
        self.parse_input_list(data_file)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[102.9801, 115.9465, 122.7717],
            std=[1., 1., 1.])

    def parse_input_list(self, data_file):
        f = open(data_file, 'r')
        old_list_sample = json.load(f)
        f.close()

        self.list_sample = []
        iterator = filter(lambda x: (len(x['anchors']) != 0) and (len(x['anchors']) <= 100), old_list_sample)
        for sample in iterator:
            self.list_sample.append(sample)

        self.num_sample = len(self.list_sample)
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # image to float
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    # Round x to the nearest multiple of p and x' >= x
    @staticmethod
    def round2nearest_multiple(x, p):
        return ((x - 1) // p + 1) * p

    def __getitem__(self, index):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError

    def _get_sub_batch(self):
        return NotImplementedError


class NovelProtoDataset(Dataset):
    def __init__(self, h5path, args):
        self.feat_dim = args.feat_dim * args.crop_height * args.crop_width
        self.data_path = h5path
        self.features = None
        self.labels = None
        self.num_sample = 0
        self.data = None
        self.args = args
        self._get_feat_data()

    def _get_feat_data(self):
        f = h5py.File(self.data_path, 'r')
        self.features = np.array(f['feature_map'])
        self.labels = np.array(f['labels'])
        self.num_sample = self.labels.size

        self.data = [dict() for i in range(self.num_sample)]
        for i in range(self.num_sample):
            self.data[i] = {'feature': self.features[i],
                            'label': self.labels[i]}

    def __getitem__(self, index):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError

    def _get_sub_batch(self):
        return NotImplementedError
