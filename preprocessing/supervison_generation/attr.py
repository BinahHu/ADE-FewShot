"""
generate original data file for attribute
Format:
[[attr_index for each sample], [..], ...]
"""
import json
import numpy as np
import os
import argparse


def generate_attr(args):
    f = open(args.attr_file, 'r')
    attr = json.load(f)
    f.close()
    f = open(args.base_set, 'r')
    base_set = json.load(f)
    f.close()
    f = open(args.base_list, 'r')
    base_list = json.load(f)
    f.close()

    attr_list = []
    for i, sample in enumerate(base_set):
        category = base_list.index(int(sample["obj"]))
        attr_list.append(attr[category])

    f = open(args.output, 'w')
    json.dump(attr_list, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr_file', default='../../data/ADE/ADE_Origin/attr.json')
    parser.add_argument('--output', default='../../data/ADE/ADE_Supervision/attr.json')
    parser.add_argument('--base_set', default='../../data/ADE/ADE_Origin/base_set.json')
    parser.add_argument('--base_list', default='../../data/ADE/ADE_Origin/base_list.json')
    parser.add_argument('--attr_num', default=159)

    args = parser.parse_args()

    generate_attr(args)