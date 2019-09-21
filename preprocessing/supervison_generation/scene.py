"""
generate original data file for hierarchy
Format:
[[[class_index for level i], ... ], [..], ...]
"""
import json
import numpy as np
import os
import argparse
import re

def generate_hierarchy(args):
    f = open(args.scene_file, 'r')
    scene = json.load(f)
    f.close()

    f = open(args.output, 'w')
    json.dump(scene, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', default='../../data/ADE/ADE_Origin/scene.json')
    parser.add_argument('--output', default='../../data/ADE/ADE_Supervision/scene.json')

    args = parser.parse_args()

    generate_hierarchy(args)