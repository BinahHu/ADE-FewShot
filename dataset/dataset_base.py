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


class BaseBaseDataset(Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # parse the input list
        self.parse_input_list(odgt, **kwargs)
        self.mode = None

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[102.9801, 115.9465, 122.7717],
            std=[1., 1., 1.])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        f = open(odgt, 'r')
        old_list_sample = json.load(f)
        f.close()

        self.list_sample = []
        iterator = filter(lambda x: len(x['anchors']) != 0, old_list_sample)
        for sample in iterator:
            self.list_sample.append(sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def random_crop(self, img, size=(224, 224), box=None):
        if box is not None:
            box = np.array(box).astype(np.int)
            # if the length of the box is larger than the size we want
            box_height = box[1, 1] - box[0, 1]
            box_width = box[1, 0] - box[0, 0]
            x_lower = max(0, min(box[0, 0], box[1, 0] - size[1]))
            x_upper = min(img.shape[1] - size[1], max(box[0, 0] + size[1], box[1, 0]) - size[1])
            y_lower = max(0, min(box[0, 1], box[1, 1] - size[0]))
            y_upper = min(img.shape[0] - size[0], max(box[0, 1] + size[0], box[1, 1]) - size[0])
            if box_height >= size[0] and box_width >= size[1]:
                y = random.randint(box[0, 1], box[1, 1] - size[0])
                x = random.randint(box[0, 0], box[1, 0] - size[1])
            elif box_height >= size[0]:
                y = random.randint(box[0, 1], box[1, 1] - size[0])
                x = random.randint(x_lower, x_upper)
            elif box_width >= size[1]:
                y = random.randint(y_lower, y_upper)
                x = random.randint(box[0, 0], box[1, 0] - size[1])
            else:
                y = random.randint(y_lower, y_upper)
                x = random.randint(x_lower, x_upper)

            result = img[y:y + size[0], x:x + size[1]]
            return [result, y, x]
        else:
            h, w, _ = img.shape
            # print("{} {}".format(h, w))
            y = random.randint(0, h - size[0])
            x = random.randint(0, w - size[1])
            result = img[y:y + size[0], x:x + size[1], :]

            return [result, y, x]

        return [result, y, x]

    def img_transform(self, img):
        # image to float
        img = img.astype(np.float32)
        if self.mode == 'train':
            random_flip = np.random.choice([0, 1])
            if random_flip == 1:
                img = cv2.flip(img, 1)
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __getitem__(self, index):
        return NotImplementedError

    def __len__(self):
        return NotImplementedError

    def _get_sub_batch(self):
        return NotImplementedError


class BaseNovelDataset(Dataset):
    def __init__(self, h5path, opt, **kwargs):
        self.feat_dim = opt.feat_dim
        self.data_path = h5path
        self.features = None
        self.labels = None
        self.num_sample = 0
        self.data = None
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
