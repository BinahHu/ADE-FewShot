"""
Transform the annotation of a single image into bounding boxes
"""
import cv2
import numpy as np
import json
import os
import sys

fgd = set(json.load(open('objlist.json')))
img_paths = json.load(open('img_path.json'))
objstat = dict()
IMGNUM = 22210

def transform_annotation(dir_path, img_id):
    """
    read the image and generate bounding box annotations into a json file
    json file is specified by [{obj:id, box:[]}]
    :param dir_path:
    :param img_path:
    :return: None
    """
    seg_path = img_paths[img_id][:-4] + "_seg.png"
    img_path = os.path.join(dir_path, seg_path)
    result = {}
    result['img'] = img_id
    if not os.path.exists(img_path):
        print(img_path)
        print(img_id)
        print('')
        result['annotation'] = []
        return result
    img = cv2.imread(img_path)
    img = np.transpose(img, (2, 0, 1)).astype(np.int)
    [_, G, R] = img
    seg_maps = ((R/10 * 256) + G).astype(np.int)
    annotation = search_object(seg_maps)
    result['annotation'] = annotation
    return result


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
            if int(seg_map[h, w]) not in fgd:
                continue
            visiting_queue.append([h, w])
            left = right = w
            up = down = h
            while visiting_queue:
                cur_position = visiting_queue.pop()
                if seg_map[cur_position[0], cur_position[1]] == -1:
                    continue
                cur_obj = seg_map[cur_position[0], cur_position[1]]
                if int(cur_obj) not in fgd:
                    continue
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
            if sample_area <= min(100, area / 900):
                continue
            if (right - left) / (down - up) < 10 and (right - left) / (down - up) > 1 / 10:
                annotation_list.append({'obj': int(cur_obj), 'box': [left, right, up, down]})
                cur_obj = int(cur_obj)
                if cur_obj not in objstat:
                    objstat[cur_obj] = 0
                objstat[cur_obj] += 1

    return annotation_list

if __name__ == '__main__':
    res = []
    dir = "../../"
    for i in range(IMGNUM):
        res.append(transform_annotation(dir, i))

    json.dump(res, open('result.json', 'w'))
    json.dump(objstat, open('stat.json', 'w'))

