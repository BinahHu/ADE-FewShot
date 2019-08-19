import json
import numpy as np
import argparse
import os
import random
import math
from preprocessing.addcontext import add_context

def base_generation(args):
    origin_path = os.path.join(args.root_dataset, args.origin_dataset)
    supervision_path = os.path.join(args.root_dataset, args.supervision_dataset)
    base_set_path = os.path.join(origin_path, 'base_set.json')
    img_path_path = os.path.join(origin_path, 'img_path.json')
    img_path2size_path = os.path.join(origin_path, 'img_path2size.json')
    base_list_path = os.path.join(origin_path, 'base_list.json')
    f = open(base_set_path, 'r')
    base_set = json.load(f)
    f.close()
    f = open(img_path_path, 'r')
    img_path = json.load(f)
    f.close()
    f = open(img_path2size_path, 'r')
    img_path2size = json.load(f)
    f.close()
    f = open(base_list_path, 'r')
    base_list = json.load(f)
    f.close()

    # get other supervision
    supervision_path_names = []
    for supervision_src in args.supervision_src:
        supervision_path_names.append({'name': supervision_src['name'], 'path': supervision_src['path'],
                                       'type': supervision_src['type']})
    supervision_contents = []
    for supervision in supervision_path_names:
        path = os.path.join(supervision_path, supervision['path'])
        f = open(path, 'r')
        data = json.load(f)
        supervision_contents.append({'data': data, 'name': supervision['name'], 'type': supervision['type']})
        f.close()

    # initialize the sample list,add image level information
    sample_list_train = [dict() for i in range(len(img_path))]
    for i in range(len(img_path)):
        sample_list_train[i]['fpath_img'] = img_path[i]
        sample_list_train[i]['height'], sample_list_train[i]['width'] = \
            img_path2size[img_path[i]]
        sample_list_train[i]['index'] = i
        sample_list_train[i]['anchors'] = []
        # add image level information
        for j in range(len(supervision_contents)):
            if supervision_contents[j]['type'] == 'img':
                sample_list_train[i][supervision_contents[j]['name']] = supervision_contents[j]['data']

    sample_list_val = [dict() for i in range(len(img_path))]
    for i in range(len(img_path)):
        sample_list_val[i]['fpath_img'] = img_path[i]
        sample_list_val[i]['height'], sample_list_val[i]['width'] = \
            img_path2size[img_path[i]]
        sample_list_val[i]['index'] = i
        sample_list_val[i]['anchors'] = []

    # get the category information to split train and val
    # add supervision information to instance level
    all_list = [[] for category in base_list]
    for i, obj in enumerate(base_set):
        img_index = int(obj["img"])
        category = base_list.index(int(obj["obj"]))
        box = obj["box"]
        path = img_path[int(obj["img"])]
        shape = img_path2size[path]
        if args.context:
            box = add_context(args, box, shape)
        annotation = {"img": img_index, "obj": category, "box": box}
        for supervision in supervision_contents:
            if supervision['type'] == 'inst':
                data = supervision['data'][i]
                annotation[supervision['name']] = data
        all_list[category].append(annotation)

    random.seed(73)
    for category in range(len(base_list)):
        if all_list[category] is []:
            continue
        random.shuffle(all_list[category])

    # split into train and val
    for i in range(len(base_list)):
        length = len(all_list[i])
        if length == 0:
            continue
        for j in range(0, math.ceil(5 * length / 6)):
            img_index = all_list[i][j]['img']
            anchor = dict()
            anchor['anchor'] = all_list[i][j]['box']
            anchor['label'] = i
            # add instance level supervision for train
            for supervision in supervision_contents:
                if supervision['type'] == 'inst':
                    anchor[supervision['name']] = all_list[i][j][supervision['name']]
            sample_list_train[img_index]['anchors'].append(anchor)

        for j in range(math.ceil(5 * length / 6), length):
            img_index = all_list[i][j]['img']
            sample_list_val[img_index]['anchors'].append({'anchor': all_list[i][j]['box'], 'label': i})

    output_path = os.path.join(args.root_dataset, args.output)
    output_train = os.path.join(output_path, 'base_img_train.json')
    f = open(output_train, 'w')
    json.dump(sample_list_train, f)
    f.close()
    output_val = os.path.join(output_path, 'base_img_val.json')
    f = open(output_val, 'w')
    json.dump(sample_list_val, f)
    f.close()


def novel_generation(args):
    origin_path = os.path.join(args.root_dataset, args.origin_dataset)
    novel_set_path = os.path.join(origin_path, 'novel_set.json')
    img_path_path = os.path.join(origin_path, 'img_path.json')
    img_path2size_path = os.path.join(origin_path, 'img_path2size.json')
    novel_list_path = os.path.join(origin_path, 'novel_val_list.json')
    f = open(novel_set_path, 'r')
    novel_set = json.load(f)
    f.close()
    f = open(img_path_path, 'r')
    img_path = json.load(f)
    f.close()
    f = open(img_path2size_path, 'r')
    img_path2size = json.load(f)
    f.close()
    f = open(novel_list_path, 'r')
    novel_list = json.load(f)
    f.close()

    # initialize the sample list
    sample_list_train = [dict() for i in range(len(img_path))]
    for i in range(len(img_path)):
        sample_list_train[i]['fpath_img'] = img_path[i]
        sample_list_train[i]['height'], sample_list_train[i]['width'] = \
            img_path2size[img_path[i]]
        sample_list_train[i]['index'] = i
        sample_list_train[i]['anchors'] = []
    sample_list_val = [dict() for i in range(len(img_path))]
    for i in range(len(img_path)):
        sample_list_val[i]['fpath_img'] = img_path[i]
        sample_list_val[i]['height'], sample_list_val[i]['width'] = \
            img_path2size[img_path[i]]
        sample_list_val[i]['index'] = i
        sample_list_val[i]['anchors'] = []

    # get the category information to split train and val
    all_list = [[] for category in novel_list]
    for obj in novel_set:
        img_index = int(obj["img"])
        if int(obj["obj"]) not in novel_list:
            continue
        category = novel_list.index(int(obj["obj"]))
        box = obj["box"]
        path = img_path[int(obj["img"])]
        shape = img_path2size[path]
        if args.context:
            box = add_context(args, box, shape)
        annotation = {"img": img_index, "obj": category, "box": box}
        all_list[category].append(annotation)

    random.seed(73)
    for category in range(len(novel_list)):
        if all_list[category] is []:
            continue
        random.shuffle(all_list[category])

    # split into train and val
    for i in range(len(novel_list)):
        length = len(all_list[i])
        if length == 0:
            continue
        for j in range(0, args.shot):
            img_index = all_list[i][j]['img']
            sample_list_train[img_index]['anchors'].append({'anchor': all_list[i][j]['box'], 'label': i})

        for j in range(args.shot, length):
            img_index = all_list[i][j]['img']
            sample_list_val[img_index]['anchors'].append({'anchor': all_list[i][j]['box'], 'label': i})

    output_path = os.path.join(args.root_dataset, args.output)
    output_train = os.path.join(output_path, 'novel_img_test_train_context.json')
    f = open(output_train, 'w')
    json.dump(sample_list_train, f)
    f.close()
    output_val = os.path.join(output_path, 'novel_img_test_val_context.json')
    f = open(output_val, 'w')
    json.dump(sample_list_val, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root_dataset', default='../data/ADE', help='data file')
    parser.add_argument('-origin_dataset', default='ADE_Origin/', help='origin dir')
    parser.add_argument('--supervision_dataset', default='ADE_Supervision/', help='supervision information')
    parser.add_argument('-part', default='Base', help='Base or Novel')
    parser.add_argument('-shot', default=5, help='shot in Novel')
    parser.add_argument('-img_size', default='img_path2size.json', help='img size file')
    parser.add_argument('--supervision_src', default=json.load(open('./supervision.json', 'r')), type=list)
    parser.add_argument('-context', type=bool, default=True)
    parser.add_argument('-ratio', type=float, default=1.5)
    # example [{'type': 'img', 'name': 'seg', 'path': '1.json'},
    # {'type': 'inst', 'name': 'attr', 'path': 'attr.json'}]

    args = parser.parse_args()
    setattr(args, 'output', 'ADE_' + args.part)
    if args.part == 'Base':
        base_generation(args)
    elif args.part == 'Novel':
        novel_generation(args)
