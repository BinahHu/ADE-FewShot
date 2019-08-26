"""
get a file for the segmentation mask
store the path of the segmentation images
"""
import json
import argparse
import os
import numpy as np
import cv2


def generate_seg(args):
    """
    get the img path, replace the tail of file name into mask type
    :param args: argument
    :return: nothing, store the segment file in the allocated path
    """
    f = open(args.img_path_file, 'r')
    img_paths = json.load(f)
    f.close()

    f = open('../../data/ADE/ADE_Origin/base_list.json')
    base_list = json.load(f)
    f.close()

    base_map = {}
    for i in range(len(base_list)):
        base_map[base_list[i]] = i
    base_set = set(base_map.keys())

    seg_paths = []
    length = len(img_paths)
    for i, img_path in enumerate(img_paths):
        seg_path_original = img_path[:-4] + '_seg.png'
        if not os.path.exists(os.path.join('../../../' + seg_path_original)):
            raise RuntimeError('{} not exists'.format(seg_path_original))
        seg_path = img_path[:-4] + '_seg_base.png'
        seg_paths.append(seg_path)

        segmentation = cv2.imread(os.path.join('../../../' + seg_path_original))
        B, G, R = np.transpose(segmentation, (2, 0, 1))
        seg_map = (G + 256 * (R / 10))
        H, W = seg_map.shape
        for h in range(H):
            for w in range(W):
                if seg_map[h, w] not in base_set:
                    seg_map[h, w] = 189
                else:
                    seg_map[h, w] = base_map[seg_map[h, w]]
        cv2.imwrite('../../../' + seg_path, seg_map.astype(np.uint8))

        if i % 1 == 0:
            print('{} / {}'.format(i, length))

    f = open(args.output, 'w')
    json.dump(seg_paths, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_file', default='../../data/ADE/ADE_Origin/img_path.json')
    parser.add_argument('--output', default='../../data/ADE/ADE_Supervision/seg.json')
    args = parser.parse_args()

    generate_seg(args)
