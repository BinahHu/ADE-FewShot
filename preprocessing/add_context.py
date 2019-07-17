"""
Add context and random crop
"""

import cv2
import json
import os
import numpy as np
import math
import random
import argpase

def main(args):
    origin_dataset = os.path.join(args.root_dataset, args.origin_dataset)
    base_set_path = os.path.join(origin_dataset, 'base_set.json')
    base_list_path = os.path.join(origin_dataset, 'base_list.json')
    img_size_path = os.path.join(origin_dataset, args.img_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParse()
    parser.add_argument('-ratio', type=float, default=4.0)
    parser.add_argument('-origin_dataset', default='ADE_Origin/')
    parser.add_argument('-root_dataset', default='../data/ADE/')
    parser.add_argument('-img_size', default='img_path2size.json')
    parser.add_argument('-output_base', default='ADE_Base_With_Context')
    parser.add_argument('-output_novel', default='ADE_Novel_With_Context')
    args = parser.parse_args()

    main(args)
