"""
get a file for the segmentation mask
store the path of the segmentation images
"""
import json
import argparse
import os


def generate_seg(args):
    """
    get the img path, replace the tail of file name into mask type
    :param args: argument
    :return: nothing, store the segment file in the allocated path
    """
    f = open(args.img_path_file, 'r')
    img_paths = json.load(f)
    f.close()

    seg_paths = []
    for img_path in img_paths:
        seg_path = img_path[:-4] + '_seg.png'
        if not os.path.exists(os.path.join('../../../' + seg_path)):
            raise RuntimeError('{} not exists'.format(seg_path))
        seg_paths.append(seg_path)

    f = open(args.output, 'w')
    json.dump(seg_paths, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_file', default='../../data/ADE/ADE_Origin/img_path.json')
    parser.add_argument('--output', default='../../data/ADE/ADE_Supervision/seg.json')
    args = parser.parse_args()

    generate_seg(args)