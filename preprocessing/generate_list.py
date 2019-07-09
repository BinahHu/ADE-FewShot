"""
Generate the list for training and testing
"""
import json
import numpy as np
import os
import argparse
import random


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
    result = ""

    for obj in base_set:
        path = img_path[int(obj["img"])]
        category = base_list.index(int(obj["obj"]))
        box = obj["box"]

        result += ('{' + '\"fpath_img\": ' + '\"' + path + '\"' + ', ')
        result += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
        result += ('\"' + 'cls_label' + '\": ' + str(category) + '}' + '\n')

    result.replace("\'", '\"')
    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_obj.odgt')
    f = open(output_path, 'w')
    f.write(result)
    f.close()


def base_img_list(args, base_set, base_list, img_path, data_img):
    """
    Generate image level base train list
    """
    result = ""

    for img in data_img:
        path = img_path[int(img["img"])]
        seg_path = path[:-4] + "_seg.png"
        annotation = img["annotation"]

        new_annotate = []
        for i, box in enumerate(annotation):
            category_id = int(box["obj"])
            anchor = box["box"]
            if category_id in base_list:
                new_box = {}
                new_box["cls_label"] = base_list.index(category_id)
                new_box["anchor"] = [[anchor[0], anchor[2]], [anchor[1], anchor[3]]]
                new_annotate.append(new_box)
        new_img = {}
        new_img["fpath_img"] = path
        new_img["fpath_segm"] = seg_path
        new_img["annotation"] = new_annotate
        result += str(new_img)
        result += "\n"
    result.replace('\'', '\"')
    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_img.odgt')
    f = open(output_path, 'w')
    f.write(result)
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
        novel_obj_list(args, novel_set, novel_list, img_path)


def novel_obj_list(args, novel_set, novel_list, img_path):
    """
    Generate image level novel train/query list
    """
    # first get all the novel obj with their information
    all_list = [[] for category in novel_list]
    for obj in novel_set:
        path = img_path[int(obj["img"])]
        category = novel_list.index(int(obj["obj"]))
        box = obj["box"]
        annotation = {"path": path, "obj":category, "box": box}
        all_list[category].append(annotation)

    # split the train & test according to the shot
    for i in range(len(novel_list)):
        random.shuffle(all_list[i])

    result_train = ""
    result_test = ""

    for i in range(len(novel_list)):
        for j in range(args.shot):
            result_train += ('{' + '\"fpath_img\": ' + '\"' + all_list[i][j]["path"] + '\"' + ', ')
            box = all_list[i][j]["box"]
            result_train += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
            result_train += ('\"' + 'cls_label' + '\": ' + str(i) + '}' + '\n')

    for i in range(len(novel_list)):
        for j in range(args.shot, len(all_list[i])):
            result_test += ('{' + '\"fpath_img\": ' + '\"' + all_list[i][j]["path"] + '\"' + ', ')
            box = all_list[i][j]["box"]
            result_test += ('\"' + 'anchor' + '\": ' + str([[box[0], box[2]], [box[1], box[3]]]) + ', ')
            result_test += ('\"' + 'cls_label' + '\": ' + str(i) + '}' + '\n')

    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'novel_obj_train.odgt')
    f = open(output_path, 'w')
    f.write(result_train)
    f.close()

    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'novel_obj_test.odgt')
    f = open(output_path, 'w')
    f.write(result_test)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dataset', default='../data/ADE/')
    parser.add_argument('-origin_dataset', default='ADE_Origin/')
    parser.add_argument('-dest', default='list')
    parser.add_argument('-part', default='novel')
    parser.add_argument('-mode', default='obj')
    parser.add_argument('-output', default='ADE_Novel/')
    parser.add_argument('-shot', default=10)
    args = parser.parse_args()

    if args.dest == 'list':
        if args.part == 'base':
            base_list(args)
        elif args.part == 'novel':
            novel_list(args)
