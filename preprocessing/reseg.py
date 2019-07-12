import cv2
import numpy as np
import json
import os
import sys

map = json.load(open('../utils/map.json'))

def reseg_img(dir_path, img_path):
    img_path = os.path.join(dir_path, img_path)
    img = cv2.imread(img_path)
    print(img_path)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            [b, g, r] = img[i][j]
            cls = int(r / 10 * 256 + g) 
            key = str(cls)
            if key not in map:
                cls = 0
            else:
                cls = map[key]

            r = cls // 256 * 10
            g = cls % 256

            img[i][j] = np.array([b, g, r])
    img_path = img_path[:-7] + "nseg.png"
    cv2.imwrite(img_path, img)
    
    print('Finish')
    return None

def reseg_dir(dir_path, key='seg'):
    root_path = dir_path
    file_names = os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, file_names[0])):
        for file_name in file_names:
            reseg_dir(os.path.join(root_path, file_name))
    else:
        for file_name in file_names:
            if key in file_name:
                reseg_img(root_path, file_name)
    return None


if __name__ == '__main__':
    train_dir = '/Users/binah/cmu/ADE20K_2016_07_26/images/training/'
    valid_dir = '/Users/binah/cmu/ADE20K_2016_07_26/images/validation/'
    dir = sys.argv[1]
    reseg_dir(os.path.join(train_dir, dir))
    reseg_dir(os.path.join(validation_dir, dir))
