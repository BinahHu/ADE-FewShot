"""
generate original data file for hierarchy
Format:
[[[class_index for level i], ... ], [..], ...]
"""
import json
import numpy as np
import os
import argparse


def generate_hierarchy(args):
    f = open(args.hierarchy_file, 'r')
    hierarchy = json.load(f)
    f.close()
    f = open(args.base_set, 'r')
    base_set = json.load(f)
    f.close()
    f = open(args.base_list, 'r')
    base_list = json.load(f)
    f.close()
    f = open(args.layer_width, 'r')
    layer_width = json.load(f)
    f.close()

    hierarchy_list = []
    for i, sample in enumerate(base_set):
        category = base_list.index(int(sample["obj"]))
        hierarchy_list.append(hierarchy[category])

    f = open(args.output, 'w')
    json.dump(hierarchy_list, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchy_file', default='../../data/ADE/ADE_Origin/hierarchy.json')
    parser.add_argument('--output', default='../../data/ADE/ADE_Supervision/hierarchy.json')
    parser.add_argument('--base_set', default='../../data/ADE/ADE_Origin/base_set.json')
    parser.add_argument('--base_list', default='../../data/ADE/ADE_Origin/base_list.json')
    parser.add_argument('--layer_width', default='../../data/ADE/ADE_Origin/layer_width.json')

    args = parser.parse_args()

    generate_hierarchy(args)