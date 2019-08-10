import json
import numpy as np
import random


if __name__ == '__main__':
    f = open('ADE_Origin/novel_list.json', 'r')
    novel_class = json.load(f)
    f.close()

    random.seed(73)
    random.shuffle(novel_class)

    novel_val = novel_class[:100]
    novel_test = novel_class[100:]

    f = open('ADE_Origin/novel_val_list.json', 'w')
    json.dump(novel_val, f)
    f.close()

    f = open('ADE_Origin/novel_test_list.json', 'w')
    json.dump(novel_test, f)
    f.close()
