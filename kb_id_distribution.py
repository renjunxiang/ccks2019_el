import json
import re
import os
import pickle
from collections import defaultdict
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split


def find_topk(x, k=1):
    v = [[i, x.count(i)] for i in sorted(list(set(x)))]
    v = sorted(v, key=lambda x: x[1], reverse=True)

    return v[:k]


if not os.path.exists('./trick/entity_info.pkl'):
    num_words = 10000

    # 读取实体词典，用于推断，根据"实体-id"检索
    with open('./data_deal/%d/alias_data.pkl' % num_words, 'rb') as f:
        alias_data = pickle.load(f)

    # 读取训练集预处理
    with open('./data_deal/%d/train_data.pkl' % num_words, 'rb') as f:
        train_data = pickle.load(f)
    # 处理训练集
    id_list = defaultdict(list)
    for idx, i in enumerate(train_data):
        entity_list = i['entity_list']
        for j in entity_list:
            id_list[j[1]].append(j[0])
    f.close()

    entity_info = {}
    for i, j in id_list.items():
        top1 = find_topk(j, k=1)[0]
        entity_info.update({
            i: {
                'id_count': len(j),
                'alias_count': len(alias_data[i]),
                'most': (top1[0], top1[1] / len(j)),
                'id_list': sorted(j)
            }
        })

    with open('./trick/entity_info.pkl', 'wb') as f:
        pickle.dump(entity_info, f)



if not os.path.exists('./trick/entity_info_train.pkl'):
    num_words = 10000

    # 读取实体词典，用于推断，根据"实体-id"检索
    with open('./data_deal/%d/alias_data.pkl' % num_words, 'rb') as f:
        alias_data = pickle.load(f)

    # 读取训练集预处理
    with open('./data_deal/%d/train_data.pkl' % num_words, 'rb') as f:
        train_data = pickle.load(f)

    # 拆分训练集
    train1_data, train2_data = train_test_split(train_data,
                                                test_size=0.1,
                                                random_state=1)

    # 读取训练集预处理
    with open('./data_deal/%d/train_data.pkl' % num_words, 'rb') as f:
        train_data = pickle.load(f)
    # 处理训练集
    id_list = defaultdict(list)
    for idx, i in enumerate(train1_data):
        entity_list = i['entity_list']
        for j in entity_list:
            id_list[j[1]].append(j[0])
    f.close()

    entity_info_train = {}
    for i, j in id_list.items():
        top1 = find_topk(j, k=1)[0]
        entity_info_train.update({
            i: {
                'id_count': len(j),
                'alias_count': len(alias_data[i]),
                'most': (top1[0], top1[1] / len(j)),
                'id_list': sorted(j)
            }
        })

    with open('./trick/entity_info_train.pkl', 'wb') as f:
        pickle.dump(entity_info_train, f)