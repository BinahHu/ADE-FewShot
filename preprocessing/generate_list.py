"""
Generate the list for training and testing
"""
import json
import numpy as np
import os
import argparse


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
    result = ''

    for obj in base_set:
        path = img_path[int(obj["img"])]
        category = base_list.index(int(obj["obj"]))
        box = obj["box"]
        new_obj = {"fpath_img": path, "obj": category,
                   "anchor": [[box[0], box[2]], [box[1], box[3]]]}
        result += str(new_obj)
        result += '\n'

    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_obj.odgt')
    f = open(output_path, 'w')
    f.write(result)
    f.close()


def base_img_list(args, base_set, base_list, img_path, data_img):
    """
    Generate image level base train list
    """
    result = ''

    for img in data_img:
        path = img_path[int(img["img"])]
        seg_path = path[:-4] + "_seg.png"
        annotation = img['annotation']

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
        result += '\n'

    output_path = os.path.join(os.path.join(args.root_dataset, args.output), 'base_img.odgt')
    f = open(output_path, 'w')
    f.write(result)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dataset', default='../data/ADE/')
    parser.add_argument('-origin_dataset', default='ADE_Origin/')
    parser.add_argument('-dest', default='list')
    parser.add_argument('-part', default='base')
    parser.add_argument('-mode', default='obj')
    parser.add_argument('-output', default='ADE_Base/')
    args = parser.parse_args()

    if args.dest == 'list':
        if args.part == 'base':
            base_list(args)
