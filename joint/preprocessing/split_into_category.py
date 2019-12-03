"""
Split the whole thing into different categories for balance sampling
"""
import argparse
import numpy as np
import h5py
import random
import os


def main(args):
    folder_path = '../data_file/{}_{}/'.format(args.model, args.target_shot)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if args.part == 'train':
        f = h5py.File(args.feat_folder + '{}_set_{}_{}.h5'.format(args.part, args.model, args.base_shot), 'r')
    elif args.part == 'tail':
        f = h5py.File(args.feat_folder + '{}_set_{}.h5'.format(args.part, args.model), 'r')
    elif args.part == 'val':
        f = h5py.File(args.feat_folder + '{}_set_{}_{}.h5'.format(args.part, args.model, args.base_shot), 'r')
    train_feat = np.array(f['feature_map'])
    train_label = np.array(f['labels']).astype(np.int)
    f.close()
    print('input complete')

    if args.target_shot != 10 or args.part != 'train':
        train_num = train_label.shape[0]
        f = h5py.File(folder_path + '{}_label.h5'.format(args.part), 'w')
        f.create_dataset('labels'.format(args.part), data=train_label)
        f.close()
        print("start to split and record")
        count = 0
        while count * 100 < train_num:
            f = h5py.File(folder_path + '{}_feat_{}.h5'.format(args.part, count), 'w')
            f.create_dataset('feature_map', data=train_feat[count*100:min((count+1)*100, train_num)])
            f.close()
            count += 1
            if count % 100 == 0:
                print('{} / {}'.format(count, train_num // 100))
    else:
        print("start re-ordering")
        train_num = train_label.shape[0]
        category_instance_sum = np.zeros(args.class_num).astype(np.int)
        for i in range(train_num):
            category_instance_sum[train_label[i]] += 1
        start_location = np.zeros(args.class_num).astype(np.int)
        for i in range(1, args.class_num):
            start_location[i] = start_location[i - 1] + category_instance_sum[i - 1]

        new_train_set = np.zeros((train_num, args.feat_dim))
        new_label_set = np.zeros(train_num).astype(np.int)
        count = np.zeros(train_num).astype(np.int)
        for i in range(train_num):
            label = train_label[i]
            location = start_location[label] + count[label]
            new_train_set[location, :] = train_feat[i, :].copy()
            new_label_set[location] = train_label[i]
            count[label] += 1

        print('start recording completed file')
        f = h5py.File(folder_path + '{}_label.h5'.format(args.part), 'w')
        f.create_dataset('labels', data=new_label_set)
        f.create_dataset('start_location', data=start_location)
        f.create_dataset('category_num', data=category_instance_sum)
        f.close()

        index = 0
        while index * 100 < train_num:
            f = h5py.File(folder_path + '{}_feat_{}.h5'.format(args.part, index), 'w')
            f.create_dataset('feature_map', data=new_train_set[index * 100:min((index + 1) * 100, train_num)])
            f.close()
            index += 1
            if index % 100 == 0:
                print('{} / {}'.format(index, train_num // 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dim', default=4608)
    parser.add_argument('--base_class_num', default=189, type=int)
    parser.add_argument('--feat_folder', default='../data/')
    parser.add_argument('--novel_class_num', default=193, type=int)

    parser.add_argument('--model', default='baseline')
    parser.add_argument('--base_shot', default=0, type=int,
                        help='base class has how many shots')
    parser.add_argument('--target_shot', default=0, type=int)
    parser.add_argument('--part', default='train', type=str)

    args = parser.parse_args()
    args.class_num = args.base_class_num + args.novel_class_num
    main(args)

