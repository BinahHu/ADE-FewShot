import os
import torch
import numpy as np
import cv2
from dataset.transform import Transform
from dataset.proto_dataset import BaseProtoDataset


class BaseDataset(BaseProtoDataset):
    """
    Form batch at object level
    """
    def __init__(self, data_file, args):
        super(BaseDataset, self).__init__(data_file, args)
        self.root_dataset = args.root_dataset
        # down sampling rate of feature map
        self.down_sampling_rate = args.down_sampling_rate
        self.batch_per_gpu = args.batch_size_per_gpu
        self.batch_record_list = [[], []]

        self.cur_idx = 0
        self.if_shuffled = False
        self.max_anchor_per_img = args.max_anchor_per_img

        self.mode = 'train'

        if hasattr(args, 'supervision'):
            self.supervision = args.supervision
        self.transform = Transform(args)

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
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
        # if not self.if_shuffled:
        #     np.random.shuffle(self.list_sample)
        #     self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        this_short_size = self.imgShortSize

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resize_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        batch_scales = np.zeros((self.batch_per_gpu, 2), np.float)
        batch_anchor_num = np.zeros(self.batch_per_gpu)
        batch_labels = np.zeros((self.batch_per_gpu, self.max_anchor_per_img))
        batch_anchors = np.zeros((self.batch_per_gpu, self.max_anchor_per_img, 4))

        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), self.imgMaxSize / max(img_height, img_width))
            img_resize_height, img_resize_width = img_height * this_scale, img_width * this_scale
            batch_resize_size[i, :] = img_resize_height, img_resize_width
        batch_resize_height = np.max(batch_resize_size[:, 0])
        batch_resize_width = np.max(batch_resize_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resize_height = int(self.round2nearest_multiple(batch_resize_height, self.padding_constant))
        batch_resize_width = int(self.round2nearest_multiple(batch_resize_width, self.padding_constant))

        for i in range(self.batch_per_gpu):
            batch_scales[i, 0] = batch_resize_height / batch_records[i]['height']
            batch_scales[i, 1] = batch_resize_width / batch_records[i]['width']

        assert self.padding_constant >= self.down_sampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resize_height, batch_resize_width)

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            assert (img.ndim == 3)
            # note that each sample within a mini batch has different scale param
            img = cv2.resize(img, (batch_resize_width, batch_resize_height), interpolation=cv2.INTER_CUBIC)
            # image transform
            img = self.img_transform(img)
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            anchors = batch_records[i]['anchors']
            anchor_num = min(len(anchors), self.max_anchor_per_img)
            batch_anchor_num[i] = anchor_num
            for j in range(anchor_num):
                label = int(anchors[j]['label'])
                batch_labels[i, j] = label
                batch_anchors[i, j, :] = np.array(anchors[j]['anchor'])

        output = dict()
        output['img_data'] = batch_images
        output['scales'] = torch.tensor(batch_scales)
        output['label'] = torch.tensor(batch_labels)
        output['anchors'] = torch.tensor(batch_anchors)
        output['anchor_num'] = torch.tensor(batch_anchor_num)

        if self.mode == 'val':
            return output

        if not hasattr(self, 'supervision'):
            return output

        # add supervision information
        for supervision in self.supervision:
            output[supervision['name']] = None

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]
            for supervision in self.supervision:
                name = supervision['name']
                # for supervision such as attr
                if supervision['type'] == 'inst':
                    # determine the shape of the supervision information
                    tensor = getattr(self.transform, name + '_transform')(this_record['anchors'][0][name],
                                                                          supervision['other'])
                    if tensor.ndim == 2 and output[name] is None:
                        output[name] = torch.zeros(self.batch_per_gpu, self.max_anchor_per_img,
                                                   tensor.shape[0], tensor.shape[1])
                    elif tensor.ndim == 1 and output[name] is None:
                        output[name] = torch.zeros(self.batch_per_gpu, self.max_anchor_per_img, tensor.shape[0])

                    for j, anchor in enumerate(this_record['anchors']):
                        content = anchor[name]
                        tensor = getattr(self.transform, name + '_transform')(content, supervision['other'])
                        output[name][i, j] = torch.from_numpy(tensor)

                elif supervision['type'] == 'img':
                    # for the supervision such as segmentation
                    if supervision['content'] == 'map':
                        img = getattr(self.transform, name + '_transform')(this_record[name], supervision['other'])
                        img = cv2.resize(img, (batch_resize_width, batch_resize_height),
                                         interpolation=cv2.INTER_NEAREST)
                        if img.ndim == 2 and output[name] is None:
                            output[name] = torch.zeros(self.batch_per_gpu, batch_resize_height, batch_resize_width)
                        elif img.ndim == 3 and output[name] is None:
                            output[name] = torch.zeros(self.batch_per_gpu, img.shape[0],
                                                       batch_resize_height, batch_resize_width)
                        output[name][i] = torch.from_numpy(img)
        return output

    def __len__(self):
        # It's a fake length due to the trick that every loader maintains its own list
        return int(1e10)
