import torch
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
import codecs
import json

DIR = '.'
num_words = 10000

# id分布
with open('./trick/entity_info_train.pkl', 'rb') as f:
    entity_info = pickle.load(f)

# 读取实体词典，用于推断，根据"实体-id"检索
with open('./data_deal/%d/alias_data.pkl' % num_words, 'rb') as f:
    alias_data = pickle.load(f)

# 读取验证集预处理
with open('./data_deal/%d/test_data722.pkl' % num_words, 'rb') as f:
    test_data = pickle.load(f)

test_predicts = []

file_name = {
    10000: {
        'bert': {
            'lstm_2_768_3_len_400_lf_2_l_2': [
                18,  # 0.743
            ],
            'lstm_3_768_3_len_400_lf_2_l_2': [
                13,  # 0.742
            ],
            'lstm_3_1024_3_len_400_lf_2_l_2': [
                14,  # 0.744
            ],
            'lstm_4_768_3_len_400_lf_2_l_2': [
                12,  # 0.743
            ],
            'lstm_4_1024_3_len_400_lf_2_l_2': [
                13,  # 0.743
            ],
            'lstm_2_768_2_len_400_lf_2_l_2': [
                17,  # 0.744
            ],
            'lstm_2_1024_2_len_400_lf_2_l_2': [
                12,  # 0.742
            ],
            'lstm_3_768_2_len_400_lf_2_l_2': [
                20,  # 0.741
            ],
            'lstm_3_1024_2_len_400_lf_2_l_2': [
                22,  # 0.741
            ],
            'lstm_4_768_2_len_400_lf_2_l_2': [
                13,  # 0.742
            ],
            'lstm_4_1024_2_len_400_lf_2_l_2': [
                15,  # 0.742
            ],
        },
        'wwm': {
            'lstm_2_768_3_len_400_lf_2_l_2': [
                17,  # 0.744
            ],
            'lstm_2_1024_3_len_400_lf_2_l_2': [
                9,  # 0.745
            ],
            'lstm_3_768_3_len_400_lf_2_l_2': [
                18,  # 0.744
            ],
            'lstm_3_1024_3_len_400_lf_2_l_2': [
                14,  # 0.742
            ],
            'lstm_4_768_3_len_400_lf_2_l_2': [
                13,  # 0.743
            ],
            'lstm_4_1024_3_len_400_lf_2_l_2': [
                14,  # 0.743
            ],
            'lstm_2_768_2_len_400_lf_2_l_2': [
                20,  # 0.744
            ],
            'lstm_2_1024_2_len_400_lf_2_l_2': [
                17,  # 0.743
            ],
            'lstm_3_768_2_len_400_lf_2_l_2': [
                21,  # 0.745
            ],
            'lstm_3_1024_2_len_400_lf_2_l_2': [
                18,  # 0.743
            ],
            'lstm_4_768_2_len_400_lf_2_l_2': [
                19,  # 0.742
            ],
            'lstm_4_1024_2_len_400_lf_2_l_2': [
                21,  # 0.743
            ],
        },
        'ernie': {
            'lstm_3_768_3_len_400_lf_2_l_2': [
                24,  # 0.743
            ],
            'lstm_4_768_3_len_400_lf_2_l_2': [
                9,  # 0.739
            ],
            'lstm_4_768_2_len_400_lf_2_l_2': [
                19,  # 0.742
            ],
            'lstm_3_768_2_len_400_lf_2_l_2': [
                7,  # 0.740
            ],
            'lstm_3_1024_2_len_400_lf_2_l_2': [
                12,  # 0.742
            ],
            'lstm_4_1024_2_len_400_lf_2_l_2': [
                21,  # 0.742
            ],
        },
    }
}

for num_words, value1 in file_name.items():
    for embedding_name, value2 in value1.items():
        for model_name, model_idxs in value2.items():
            for model_idx in model_idxs:
                with open(
                        './results_test/%d/%s/%s/test_%03d.pkl' % (num_words, embedding_name, model_name, model_idx),
                        'rb') as f:
                    test = pickle.load(f)
                    test_predicts.append(test)


def find_topk(x, k=1):
    v = [[i, x.count(i)] for i in sorted(list(set(x)))]
    v = sorted(v, key=lambda x: x[1], reverse=True)

    return v[:k]


# 补全部分实体
def guanjianci(line_ensemble_set, line_ensemble, word):
    """
    ['纪录片','小说','汉化组','宣传片','设计师','漫画',视频','电影']
    :param text:
    :param line_ensemble_set:
    :param line_ensemble:
    :param word:
    :return:
    """
    w_l = len(word)
    word_in_entity_set = [j for j in line_ensemble_set if j[1][-w_l:] == word and len(j[1]) > w_l]
    word_in_entity_list = [j for j in line_ensemble if j[1][-w_l:] == word and len(j[1]) > w_l]
    if len(word_in_entity_set) > 0 and len(word_in_entity_list) > 0:
        entity_top = find_topk(word_in_entity_list)[0][0]
        line_ensemble = line_ensemble + [entity_top] * 30

        # 移除错误
        remove_list = []
        for one in line_ensemble_set:
            if (word in one[1]) and (one[-1] == entity_top[-1]) and (len(one[1]) < len(entity_top[1])):
                remove_list.append(one)

        line_ensemble_set = [j for j in line_ensemble_set if j not in remove_list]

    return line_ensemble_set, line_ensemble


# 书名号内如果完整不拆开
def shuming(line_ensemble_set, line_ensemble, text):
    re_list = list(re.finditer('《[^《]+》', text))
    if re_list:
        remove_list = []
        for i in re_list:
            s = i.start() + 1
            e = i.end() - 2
            shuming_entity = text[s:(e + 1)]
            if shuming_entity in alias_data:
                for j in line_ensemble_set:
                    if j[2] >= s and j[3] <= e and (j[3] - j[2]) < (e - s):
                        remove_list.append(j)

        line_ensemble_set_ = [i for i in line_ensemble_set if i not in remove_list]
    else:
        line_ensemble_set_ = line_ensemble_set

    return line_ensemble_set_, line_ensemble


def test_score(i, n1=1):
    f = codecs.open('./submit_test/eval_0725.json', 'w', encoding='utf-8')
    for idx, data in enumerate(test_data):

        line_ensemble_raw = []
        for test_predict in test_predicts:
            line_ensemble_raw += test_predict[idx]

        # 有歧义、得分<0.5、语料频数>1、语料频率>0.9
        line_ensemble_raw_ = []
        for j in line_ensemble_raw:
            kb_id, entity, s, e, score = j
            k = j
            if entity in entity_info:
                entity_most = entity_info[entity]['most']
                id_count = entity_info[entity]['id_count']
                if len(alias_data[entity]) > 1 and score < 0.5 and id_count > 1:
                    if entity_most[-1] > 0.9 and entity_most[0] != kb_id and entity_most[0] != 'NIL':
                        k = (entity_info[entity]['most'][0], entity, s, e, score)
            line_ensemble_raw_.append(k)
        line_ensemble_raw = line_ensemble_raw_

        line_ensemble = [j[:-1] for j in line_ensemble_raw]
        line_ensemble_set = sorted(list(set(line_ensemble)))

        line_ensemble_set, line_ensemble = shuming(line_ensemble_set, line_ensemble, data['text'])

        for j in [
            # '纪录片', '小说', '汉化组', '宣传片',
            '设计师', '吧',
            # '艺人', '网盘', '院士',
            # '将军',
        ]:
            line_ensemble_set, line_ensemble = guanjianci(line_ensemble_set,
                                                          line_ensemble,
                                                          j)

        line_ensemble_new = [j for j in line_ensemble_set if line_ensemble.count(j) > i]

        # 规则 补上range(i - n1, i + 1)
        line_ensemble_new_ = [j for j in line_ensemble_set if
                              line_ensemble.count(j) in range(i - n1, i + 1) and len(alias_data[j[1]]) == 1]
        line_ensemble_new = line_ensemble_new + line_ensemble_new_

        mention_data = []
        for j in line_ensemble_new:
            mention_data.append({
                "kb_id": j[0],
                "mention": j[1],
                "offset": str(j[2])
            })

        line_result = {
            "text_id": data['text_id'],
            "text": data['text'],
            "mention_data": mention_data,
            "dev_or_eval": "dev" if data['text'] in cheat else "eval"
        }
        f.write(json.dumps(line_result, ensure_ascii=False) + '\n')
    f.close()

    print('模型数量：', len(test_predicts))


if __name__ == '__main__':
    test_score(13, n1=5)
