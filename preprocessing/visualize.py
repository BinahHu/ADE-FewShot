import cv2
import numpy as np
import os
import json


def visualize_box_img(img_path):
    """
    visualize the bounding boxes of an image
    :param img_path: image path
    :return: None
    """
    json_path = img_path[:-3] + 'json'
    if not os.path.exists(json_path):
        return None
    f = open(json_path, 'r')
    annotations = json.load(f)
    img = cv2.imread(img_path).astype(np.uint8)
    for box in annotations:
        left, right, up, down = box['box']
        img = cv2.rectangle(img, (int(left), int(up)), (int(right), int(down)), (255, 255, 255), 4)

    out_path = img_path[:-4] + '_box.jpg'
    cv2.imwrite(out_path, img)


def visualize_box_dir(dir_path):
    root_path = dir_path
    file_names = os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, file_names[0])):
        for file_name in file_names:
            visualize_box_dir(os.path.join(root_path, file_name))
    else:
        for file_name in file_names:
            if 'box' not in file_name and 'jpg' in file_name:
                visualize_box_img(os.path.join(root_path, file_name))

if __name__ == '__main__':
    visualize_box_dir('/Users/binah/cmu/ADE20K_2016_07_26/images/training/a/aquarium')
