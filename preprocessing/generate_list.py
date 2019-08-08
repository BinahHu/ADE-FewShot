"""
Add cluster mask to imgs
"""
import json
import numpy as np
import os
import argparse
import random
import math
random.seed(73)


def base_list(args):
    origin_dataset = os.path.join(args.root_dataset, args.origin_dataset)
    base_set_path = os.path.join(origin_dataset, 'base_set.json')
    base_list_path = os.path.join(origin_dataset, 'base_list.json')
    img_path_path = os.path.join(origin_dataset, 'img_path.json')

    f = open(base_set_path, 'r')
    base_set = json.load(f)
    f.close()
    f = open(base_list_path, 'r')
    base_list = json.load(f)
    f.close()
    f = open(img_path_path, 'r')
    img_path = json.load(f)
    f.close()

    result_train = []
    result_val = []
    all_list = [[] for category in base_list]
    mask = args.maskset

    for obj in base_set:
        path = img_path[int(obj["img"])]
        category = base_list.index(int(obj["obj"]))
        if args.mask and str(category) in mask:
            category  = mask[str(category)]
        box = obj["box"]
        annotation = {"path": path, "obj": category, "box": box}
        all_list[category].append(annotation)

    random.seed(73)
    for category in range(len(base_list)):
        if all_list[category] is []:
            continue
        random.shuffle(all_list[category])

    for i in range(len(base_list)):
        length = len(all_list[i])
        if length == 0:
            continue
        train_num = math.ceil(5 * length / 6)
        for j in range(0, train_num):
            sample = dict()
            sample['fpath_img'] = all_list[i][j]["path"]
            sample['anchor'] = all_list['box']
            sample['label'] = i
            result_train.append(sample)

        for j in range(train_num, length):
            sample = dict()
            sample['fpath_img'] = all_list[i][j]["path"]
            sample['anchor'] = all_list[i][j]['box']
            sample['label'] = i
            result_val.append(sample)
    
    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_obj_train.json')
    f = open(output_path, 'w')
    json.dump(result_train, f)
    f.close()
    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_obj_val.json')
    f = open(output_path, 'w')
    json.dump(result_val, f)
    f.close()


def novel_list(args):
    original_dataset = os.path.join(args.root_dataset, args.origin_dataset)
    novel_set_path = os.path.join(original_dataset, 'novel_set.json')
    novel_list_path = os.path.join(original_dataset, 'novel_list.json')
    img_path_path = os.path.join(original_dataset, 'img_path.json')

    f = open(novel_set_path, 'r')
    novel_set = json.load(f)
    f.close()
    f = open(novel_list_path, 'r')
    novel_list = json.load(f)
    f.close()
    f = open(img_path_path, 'r')
    img_path = json.load(f)
    f.close()

    img_size_path = os.path.join(os.path.join(args.root_dataset, args.origin_dataset), args.img_size)
    result_train = []
    result_val = []
    all_list = [[] for category in novel_list]

    for obj in novel_set:
        path = img_path[int(obj["img"])]
        category = novel_list.index(int(obj["obj"]))
        box = obj["box"]
        annotation = {"path": path, "obj": category, "box": box}
        all_list[category].append(annotation)

    random.seed(73)
    for category in range(len(novel_list)):
        random.shuffle(all_list[category])

    for i in range(len(novel_list)):
        for j in range(0, args.shot):
            sample = dict()
            sample['fpath_img'] = all_list[i][j]["path"]
            sample['anchor'] = all_list['box']
            sample['label'] = i
            result_train.append(sample)

        for j in range(args.shot, len(all_list[i])):
            sample = dict()
            sample['fpath_img'] = all_list[i][j]["path"]
            sample['anchor'] = all_list[i][j]['box']
            sample['label'] = i
            result_val.append(sample)

    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'novel_obj_train.json')
    f = open(output_path, 'w')
    f.write(result_train)
    f.close()
    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'novel_obj_val.json')
    f = open(output_path, 'w')
    f.write(result_val)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dataset', default='../data/ADE/')
    parser.add_argument('-origin_dataset', default='ADE_Origin/')
    parser.add_argument('-part', default='Base')
    parser.add_argument('-shot', default=5)
    parser.add_argument('-img_size', default='img_path2size.json')
    args = parser.parse_args()
    setattr(args, 'output', 'ADE_' + args.part)

    if args.part == 'base':
        base_list(args)
    elif args.part == 'novel':
        novel_list(args)
