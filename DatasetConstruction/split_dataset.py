import json
import os

if __name__ == '__main__':
    data = json.load(open('data/data_all.json'))
    train_set = []
    test_set = []
    train_list = set(json.load(open('stat/base_list.json'))) 
    test_list = set(json.load(open('stat/novel_list.json'))) 
    for item in data:
        if item['obj'] in train_list:
            train_set.append(item)
        elif item['obj'] in test_list:
            test_set.append(item)
        else:
            #class size < 15
            continue
    json.dump(train_set, open("data/base_set.json", 'w'))
    json.dump(test_set, open("data/novel_set.json", 'w'))
