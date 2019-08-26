"""
generate things for the foreground and background segmentation
"""
import json
import argparse
import os
import cv2
import numpy as np


def generate_fgbg(args):
    """
    this function has two utilities:
    * generate the new foreground mask
    * generate the same file as segmentation
    :param args: argument
    :return: nothing, but save the images to the place and save a json
    """
    f = open(args.img_path_file, 'r')
    img_paths = json.load(f)
    f.close()
    f = open(args.base_class_list, 'r')
    base_class_list = json.load(f)
    f.close()
    f = open(args.novel_class_list, 'r')
    novel_class_list = json.load(f)
    f.close()

    fgbg_paths = []
    length = len(img_paths)
    for index, img_path in enumerate(img_paths):
        seg_path = img_path[:-4] + '_seg.png'
        if not os.path.exists(os.path.join('../../../' + seg_path)):
            raise RuntimeError('{} not exists'.format(seg_path))

        segmentation = cv2.imread(os.path.join('../../../' + seg_path))
        B, G, R = np.transpose(segmentation, (2, 0, 1))
        seg_map = (G + 256 * (R / 10)).astype(np.int)

        fgbg_map = np.where(np.isin(seg_map, base_class_list), 255, 0)
        fgbg_map += np.where(np.isin(seg_map, novel_class_list), 130, 0)

        fgbg_path = img_path[:-4] + '_fgbg.png'
        cv2.imwrite(os.path.join('../../../' + fgbg_path), fgbg_map)
        fgbg_paths.append(fgbg_path)

        if index % 10 == 0:
            print('{} / {}'.format(index, length))

    f =open(args.output, 'w')
    json.dump(fgbg_paths, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_file', default='../../data/ADE/ADE_Origin/img_path.json')
    parser.add_argument('--output', default='../../data/ADE/ADE_Supervision/fgbg.json')
    parser.add_argument('--base_class_list', default='../../data/ADE/ADE_Origin/base_list.json')
    parser.add_argument('--novel_class_list', default='../../data/ADE/ADE_Origin/novel_list.json')
    args = parser.parse_args()

    generate_fgbg(args)
