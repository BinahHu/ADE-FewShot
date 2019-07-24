import cv2
import json
import os
import argparse

img_paths  = json.load(open("../data/ADE/ADE_Origin/img_path.json"))
root_path = "../../"

cNum = 0
cMax = 256 * 256 * 256


def getcolor():
    cNum = (cNum + 100) % cMax
    r = cNum % 256
    g = (cNum // 256) % 256
    b = cNum // (256 * 256)
    return [r, g, b]
    

def show(img=None, name=None, boxes=None, colors=None):
    if img == None:
        return None
    if isinstance(img, int):
        img_path = img_paths[img]
    elif isinstance(img, str):
        img_path = img
    else:
        print("Type error!")
        return None
    full_path = os.path.join(root_path, img_path)
    if not os.path.exists(full_path):
        print("Img not exist!")
        return None
    img = cv2.imread(full_path)
    for i in range(len(boxes)):
        box = boxes[i]
        if colors == None or i >= len(colors):
            color = getcolor()
        else:
            color = colors[i]


    

def pack():
    return os.system("zip pic.zip pic/*")
