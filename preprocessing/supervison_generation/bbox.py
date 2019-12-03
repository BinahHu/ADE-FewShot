"""
generate original data file for bounding boxes
Format:
[[anchor for each sample], [..], ...]
"""
import json
import numpy as np
import os
import argparse


def generate_bbox(args):
    f = open(args.base_set, 'r')
    base_set = json.load(f)
    f.close()

    bbox_list = []
    for i, sample in enumerate(base_set):
        bbox_list.append(sample['box'])
    f = open(args.output, 'w')
    json.dump(bbox_list, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_set', default='../../data/ADE/ADE_Origin/base_set.json')
    parser.add_argument('--base_list', default='../../data/ADE/ADE_Origin/base_list.json')
    parser.add_argument('--output', default='../../data/ADE/ADE_Supervision/bbox.json')

    args = parser.parse_args()

    generate_bbox(args)
