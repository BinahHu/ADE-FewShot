"""
Transform the annotation of a single image into bounding boxes
"""
import cv2
import numpy as np
import json
import os


def transform_annotation(dir_path, img_path):
    """
    read the image and generate bounding box annotations into a json file
    json file is specified by [{obj:id, box:[]}]
    :param dir_path:
    :param img_path:
    :return: None
    """
    img_path = os.path.join(dir_path, img_path)
    img = cv2.imread(img_path)
    print(img_path)
    img = np.transpose(img, (2, 0, 1)).astype(np.int)
    [R, G, _] = img
    seg_maps = ((R/10 * 256) + G).astype(np.int)
    annotation = search_object(seg_maps)
    json_path = os.path.join(dir_path, img_path[:-8] + '.json')
    f = open(json_path, 'w')
    json.dump(annotation, f)
    print('Finish')
    return None


def search_object(seg_map):
    """
    search for objects in the map
    format is specified by [{obj:id, box:[]}]
    :param seg_map: annotation map
    :return: a list of objects with their locations
    """
    visiting_queue = []
    directions = [[0, -1], [0, 1], [1, 0], [-1, 0]]
    annotation_list = []
    cur_obj = -1
    H, W = seg_map.shape
    area = H * W

    for h in range(H):
        for w in range(W):
            if seg_map[h, w] == -1:
                continue
            visiting_queue.append([h, w])
            left = right = w
            up = down = h
            while visiting_queue:
                cur_position = visiting_queue.pop()
                if seg_map[cur_position[0], cur_position[1]] == -1:
                    continue
                cur_obj = seg_map[cur_position[0], cur_position[1]]
                seg_map[cur_position[0], cur_position[1]] = -1

                left = min(left, cur_position[1])
                right = max(right, cur_position[1])
                up = min(up, cur_position[0])
                down = max(down, cur_position[0])

                for direction in directions:
                    new_position = [cur_position[0] + direction[0], cur_position[1] + direction[1]]
                    if new_position[0] >= 0 and new_position[0] < H and new_position[1] >= 0 and new_position[1] < W:
                        if seg_map[new_position[0], new_position[1]] == cur_obj:
                            visiting_queue.append(new_position)
            sample_area = (right - left) * (down - up)
            if sample_area >= area / 3 or sample_area <= max(10, area / 10000):
                continue
            if (right - left) / (down - up) < 10 and (right - left) / (down - up) > 1 / 10:
                annotation_list.append({'obj': int(cur_obj), 'box': [left, right, up, down]})

    return annotation_list


def transform_dir(dir_path, key='seg'):
    """
    adding directory level annotations
    :param dir_path: directory path
    :param key: keywords in the segmentation annotation of the image
    :return: None
    """
    root_path = dir_path
    file_names = os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, file_names[0])):
        for file_name in file_names:
            transform_dir(os.path.join(root_path, file_name))
    else:
        for file_name in file_names:
            if key in file_name:
                transform_annotation(root_path, file_name)
    return None


if __name__ == '__main__':
    transform_dir('/home/pzq/ADE20K_2016_07_26/images/training/e')
    # transform_annotation('/home/pzq/ADE20K_2016_07_26/images/training/a/abbey', 'ADE_train_00000970_seg.png')
