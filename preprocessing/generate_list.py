"""
Generate the list for training and testing
"""
import json
import numpy as np
import os
import argparse
import random
import math


def base_list(args):
    origin_dataset = os.path.join(args.root_dataset, args.origin_dataset)
    base_set_path = os.path.join(origin_dataset, 'base_set.json')
    base_list_path = os.path.join(origin_dataset, 'base_list.json')
    img_path_path = os.path.join(origin_dataset, 'img_path.json')
    data_img_path = os.path.join(origin_dataset, 'data_img.json')

    f = open(base_set_path, 'r')
    base_set = json.load(f)
    f.close()
    f = open(base_list_path, 'r')
    base_list = json.load(f)
    f.close()
    f = open(img_path_path, 'r')
    img_path = json.load(f)
    f.close()
    f = open(data_img_path, 'r')
    data_img = json.load(f)
    f.close()

    if args.mode == 'obj':
        base_obj_list(args, base_set, base_list, img_path)
    elif args.mode == 'img':
        base_img_list(args, base_set, base_list, img_path, data_img)


def base_obj_list(args, base_set, base_list, img_path):
    """
    Generate object level base training dataset odgt
    """
    result_train = ""
    result_val = ""
    all_list = [[] for category in base_list]

    for obj in base_set:
        path = img_path[int(obj["img"])]
        category = base_list.index(int(obj["obj"]))
        box = obj["box"]
        annotation = {"path": path, "obj": category, "box": box}
        all_list[category].append(annotation)

    for category in range(len(base_list)):
        random.shuffle(all_list[category])

    for i in range(len(base_list)):
        length = len(all_list[i])
        train_num = length
        if args.cap != 0:
            train_num = min(args.cap, math.ceil(5 * train_num / 6))
        for j in range(0, train_num):
            result_train += ('{' + '\"fpath_img\": ' + '\"' + all_list[i][j]["path"] + '\"' + ', ')
            box = all_list[i][j]["box"]
            result_train += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
            result_train += ('\"' + 'cls_label' + '\": ' + str(i) + '}' + '\n')

    for i in range(len(base_list)):
        length = len(all_list[i])
        val_num = length
        if args.cap != 0:
            val_num = min(args.cap, math.ceil(5 * val_num / 6))
        for j in range(val_num, length):
            result_val += ('{' + '\"fpath_img\": ' + '\"' + all_list[i][j]["path"] + '\"' + ', ')
            box = all_list[i][j]["box"]
            result_val += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
            result_val += ('\"' + 'cls_label' + '\": ' + str(i) + '}' + '\n')

    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_obj_train_cap.odgt')
    f = open(output_path, 'w')
    f.write(result_train)
    f.close()
    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_obj_val_cap.odgt')
    f = open(output_path, 'w')
    f.write(result_val)
    f.close()


def base_img_list(args, base_set, base_list, img_path, data_img):
    """
    Generate object level base training dataset odgt
    """
    img_size_path = os.path.join(os.path.join(args.root_dataset, args.origin_dataset),
                                 args.img_size)
    f = open(img_size_path, 'r')
    image_size = json.load(f)
    f.close()

    result_train = ""
    result_val = ""
    all_list = [[] for category in base_list]

    for obj in base_set:
        path = img_path[int(obj["img"])]
        category = base_list.index(int(obj["obj"]))
        box = obj["box"]
        annotation = {"path": path, "obj": category, "box": box}
        all_list[category].append(annotation)

    for category in range(len(base_list)):
        random.shuffle(all_list[category])

    for i in range(len(base_list)):
        length = len(all_list[i])
        cap = 0
        if args.cap != 0:
            cap = min(args.cap, math.ceil(length / 6))
        else:
            cap = math.ceil(length / 6)
        for j in range(0, math.ceil(length / 6)):
            result_val += ('{' + '\"fpath_img\": ' + '\"' + all_list[i][j]["path"] + '\"' + ', ')
            box = all_list[i][j]["box"]
            result_val += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
            result_val += ('\"' + 'cls_label' + '\": ' + str(i) + ', ')
            size = image_size[all_list[i][j]['path']]
            result_val += ('\"' + 'height' + '\": ' + str(size[0]) + ', ')
            result_val += ('\"' + 'width' + '\": ' + str(size[1]) + '}' + '\n')

    for i in range(len(base_list)):
        length = len(all_list[i])
        for j in range(math.ceil(length / 6), length):
            result_train += ('{' + '\"fpath_img\": ' + '\"' + all_list[i][j]["path"] + '\"' + ', ')
            box = all_list[i][j]["box"]
            result_train += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
            result_train += ('\"' + 'cls_label' + '\": ' + str(i) + ', ')
            size = image_size[all_list[i][j]['path']]
            result_train += ('\"' + 'height' + '\": ' + str(size[0]) + ', ')
            result_train += ('\"' + 'width' + '\": ' + str(size[1]) + '}' + '\n')

    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_img_train.odgt')
    f = open(output_path, 'w')
    f.write(result_train)
    f.close()
    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_img_val.odgt')
    f = open(output_path, 'w')
    f.write(result_val)
    f.close()


def novel_list(args):
    original_dataset = os.path.join(args.root_dataset, args.origin_dataset)
    novel_set_path = os.path.join(original_dataset, 'novel_set.json')
    novel_list_path = os.path.join(original_dataset, 'novel_list.json')
    img_path_path = os.path.join(original_dataset, 'img_path.json')
    data_img_path = os.path.join(original_dataset, 'data_img.json')

    f = open(novel_set_path, 'r')
    novel_set = json.load(f)
    f.close()
    f = open(novel_list_path, 'r')
    novel_list = json.load(f)
    f.close()
    f = open(img_path_path, 'r')
    img_path  =json.load(f)
    f.close()
    f = open(data_img_path, 'r')
    data_img = json.load(f)
    f.close()

    if args.mode == 'obj':
        novel_obj_list_before_feat(args, novel_set, novel_list, img_path)


def novel_obj_list_before_feat(args, novel_set, novel_list, img_path):
    """
    Generate object level base training dataset odgt
    """
    result_train = ""
    result_val = ""
    all_list = [[] for category in novel_list]

    for obj in novel_set:
        path = img_path[int(obj["img"])]
        category = novel_list.index(int(obj["obj"]))
        box = obj["box"]
        annotation = {"path": path, "obj": category, "box": box}
        all_list[category].append(annotation)

    for category in range(len(novel_list)):
        random.shuffle(all_list[category])

    for i in range(len(novel_list)):
        for j in range(0, args.shot):
            result_train += ('{' + '\"fpath_img\": ' + '\"' + all_list[i][j]["path"] + '\"' + ', ')
            box = all_list[i][j]["box"]
            result_train += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
            result_train += ('\"' + 'cls_label' + '\": ' + str(i) + '}' + '\n')

    for i in range(len(novel_list)):
        length = len(all_list[i])
        for j in range(args.shot, length):
            result_val += ('{' + '\"fpath_img\": ' + '\"' + all_list[i][j]["path"] + '\"' + ', ')
            box = all_list[i][j]["box"]
            result_val += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
            result_val += ('\"' + 'cls_label' + '\": ' + str(i) + '}' + '\n')

    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'novel_obj_train_before_feat.odgt')
    f = open(output_path, 'w')
    f.write(result_train)
    f.close()
    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'novel_obj_val_before_feat.odgt')
    f = open(output_path, 'w')
    f.write(result_val)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dataset', default='../data/ADE/')
    parser.add_argument('-origin_dataset', default='ADE_Origin/')
    parser.add_argument('-dest', default='list')
    parser.add_argument('-part', default='base')
    parser.add_argument('-mode', default='obj')
    parser.add_argument('-output', default='ADE_Base/')
    parser.add_argument('-shot', default=5)
    parser.add_argument('-img_size', default='img_path2size.json')
    parser.add_argument('--cap', type=int, default=0)
    args = parser.parse_args()

    if args.dest == 'list':
        if args.part == 'base':
            base_list(args)
        elif args.part == 'novel':
            novel_list(args)
