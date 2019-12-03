"""
Generate train list and test list according to threshold
"""
import json
import numpy as np
import argparse
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-low', default=15, help='lowest occurrences')
    parser.add_argument('-high', default=100000, help='highest occurrences')
    parser.add_argument('-threshold', default=100, help='threshold to split')
    parser.add_argument('-sorted_index', default='stat/sortedID.json')
    parser.add_argument('-stat', default='stat/occurrence.json', help='occurrences')
    parser.add_argument('-mode', default='threshold', help='mode of split')
    args = parser.parse_args()

    f = open(args.sorted_index)
    index_list = json.load(f)
    f.close()

    f = open(args.stat)
    stat = json.load(f)
    f.close()

    stat = sorted(stat.items(), key=lambda x: x[1])
    stat_dict = defaultdict(list)
    for k, v in stat:
        stat_dict[int(k)] = int(v)

    if args.mode == 'threshold':
        train_list = []
        test_list = []
        for sample in stat_dict.items():
            if sample[1] >= args.low and sample[1] < int(args.threshold):
                test_list.append(int(sample[0]))
            elif sample[1] >= int(args.threshold) and sample[1] < args.high:
                train_list.append(int(sample[0]))

        f = open('stat/base_list.json', 'w')
        json.dump(train_list, f)
        f.close()

        f = open('stat/novel_list.json', 'w')
        json.dump(test_list, f)
        f.close()
