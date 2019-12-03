import argparse
import numpy as np 
import h5py
import random
import os


def merge_train(args):
    # first start from train
    result_feature = np.zeros((240000, args.feat_dim))
    result_label = np.zeros(240000)
    flag = 0
    for i in range(40):
        print('Train {}'.format(i))
        if os.path.exists(args.base_feat_folder + 
            'img_train_feat_{}_{}.h5'.format(args.model, i)):
            f = h5py.File(args.base_feat_folder + 'img_train_feat_{}_{}.h5'.format(args.model, i), 'r')
            base_train_feat = np.array(f['feature_map'])
            base_train_label = np.array(f['labels']).astype(np.int)
            f.close()

            result_feature[flag:flag+base_train_feat.shape[0], :] = base_train_feat
            result_label[flag:flag+base_train_feat.shape[0]] = base_train_label
            flag += base_train_feat.shape[0]
        else:
            break
    return [result_feature, result_label]


def merge_val(args):
    # first start from train
    result_feature = np.zeros((50000, args.feat_dim))
    result_label = np.zeros(50000)
    flag = 0
    for i in range(40):
        print('Val {}'.format(i))
        if os.path.exists(args.base_feat_folder + 
            'img_val_feat_{}_{}.h5'.format(args.model, i)):
            f = h5py.File(args.base_feat_folder + 'img_val_feat_{}_{}.h5'.format(args.model, i), 'r')
            base_val_feat = np.array(f['feature_map'])
            base_val_label = np.array(f['labels']).astype(np.int)
            f.close()

            result_feature[flag:flag+base_val_feat.shape[0], :] = base_val_feat
            result_label[flag:flag+base_val_feat.shape[0]] = base_val_label
            flag += base_val_feat.shape[0]
        else:
            break
    return [result_feature, result_label]
    

def main(args):
    base_train_feat, base_train_label = merge_train(args)
    base_val_feat, base_val_label = merge_val(args)

    f = h5py.File(args.novel_feat_folder + 
        'img_test_train_feat_{}.h5'.format(args.model), 'r')
    novel_train_feat = np.array(f['feature_map'])
    novel_train_label = np.array(f['labels']).astype(np.int)
    f.close()

    f = h5py.File(args.novel_feat_folder + 
        'img_test_val_feat_{}.h5'.format(args.model), 'r')
    novel_val_feat = np.array(f['feature_map'])
    novel_val_label = np.array(f['labels']).astype(np.int)
    f.close()

    print('input complete')

    base_train_num = base_train_label.shape[0]
    base_val_num = base_val_label.shape[0]
    novel_train_num = novel_train_label.shape[0]
    novel_val_num = novel_val_label.shape[0]

    # update the base label, append behind the novel 
    base_train_label += args.novel_class_num
    base_val_label += args.novel_class_num

    # base classes has limited shots
    if args.base_shot != 0:
        print('processing for base shot {}'.format(args.base_shot))
        base_train_shot_feat = np.zeros((args.base_shot * args.base_class_num, args.feat_dim))
        base_train_shot_label = np.zeros(args.base_shot * args.base_class_num).astype(np.int)

        location = [[] for i in range(args.base_class_num)]
        for i in range(base_train_num):
            location[base_train_label[i] - args.novel_class_num].append(i)

        count = 0
        for i in range(args.base_class_num):
            random.seed(10)
            random.shuffle(location[i])
            for j in range(args.base_shot):
                base_train_shot_feat[count] = base_train_feat[location[i][j]].copy()
                base_train_shot_label[count] = base_train_label[location[i][j]]
                count += 1
        del base_train_feat
        del base_train_label
        base_train_feat = base_train_shot_feat
        base_train_label = base_train_shot_label

    # naive combine
    print('began to combine the datasets')
    train_set_feat = np.concatenate((base_train_feat, novel_train_feat))
    val_set_feat = np.concatenate((base_val_feat, novel_val_feat))
    train_set_label = np.concatenate((base_train_label, novel_train_label))
    val_set_label = np.concatenate((base_val_label, novel_val_label))

    tail_set_feat = novel_val_feat
    tail_set_label = novel_val_label

    # create dataset
    print('start recording completed file')
    f = h5py.File('../data/train_set_{}_{}.h5'.format(args.model, args.base_shot), 'w')
    f.create_dataset('feature_map', data=train_set_feat)
    f.create_dataset('labels', data=train_set_label)
    f.close()
    print('training set has {} samples'.format(train_set_label.shape[0]))

    f = h5py.File('../data/val_set_{}_{}.h5'.format(args.model, args.base_shot), 'w')
    f.create_dataset('feature_map', data=val_set_feat)
    f.create_dataset('labels', data=val_set_label)
    f.close()
    print('validation set has {} samples'.format(val_set_label.shape[0]))
    
    f = h5py.File('../data/tail_set_{}.h5'.format(args.model), 'w')
    f.create_dataset('feature_map', data=tail_set_feat)
    f.create_dataset('labels', data=tail_set_label)
    f.close()
    print('tail set has {} samples'.format(tail_set_label.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dim', default=4608)
    parser.add_argument('--base_class_num', default=189, type=int)
    parser.add_argument('--base_feat_folder', default='../../data/base_feat/')
    parser.add_argument('--novel_class_num', default=193, type=int)
    parser.add_argument('--novel_feat_folder', default='../../data/base_feat/')

    parser.add_argument('--model', default='baseline')
    parser.add_argument('--base_shot', default=0, type=int, 
        help='base class has how many shots')

    args = parser.parse_args()
    main(args)
