import json
import re
import os
import pickle
from collections import defaultdict
from copy import deepcopy
import numpy as np
from keras.preprocessing.text import Tokenizer

"""
训练集
{
    'text_id': '90000',
    'text': '电影《尹灵芝》开拍',
    'entity_start': [0, 3],
    'entity_end': [1, 5],
    'entity_list': [('148097', '电影', 0, 1), ('226769', '尹灵芝', 3, 5)],
    'text_seq': [38, 47, 9, 1757, 540, 1754, 10, 191, 739]
}

知识库
{
    '七里香蔷薇': {
        '146813': {
            'type': [27],
            'data': '赵家垴妇救会副主任。汉族。中国共产党员。人物。烈士。',
            'data_seq': [14, 52, 4, 7, 50, 1, 582, 59, 7076, 1200, 1039, 71, 214, 54]
        },
        '226769': {
            'type': [33],
            'data': '尹灵芝年仅16岁就光荣的加入了中国共产党，不幸被铺。2015年中国拍摄电影。人物。',
            'data_seq': [26, 8, 12, 13, 739, 681, 38, 47, 1, 11, 27, 1, 133, 39, 11, 27, 1]
        }
    }
}
"""

num_words = 10000
texts = []
if not os.path.exists('./data_deal/%d' % num_words):
    os.mkdir('./data_deal/%d' % num_words)

# 处理训练集
f = open('./data/train.json', 'r', encoding='utf-8')
train_data = []
for idx, i in enumerate(f):
    data_one = json.loads(i)
    text_id = data_one['text_id']
    text = data_one['text'].lower()
    mention_data = data_one['mention_data']
    entity_start = []
    entity_end = []
    entity_list = []
    for mention_data_one in mention_data:
        kb_id = mention_data_one['kb_id']
        if kb_id == "NIL":
            continue
        entity = mention_data_one['mention'].lower()
        s = int(mention_data_one['offset'])
        e = s + len(entity) - 1
        entity_start.append(s)
        entity_end.append(e)
        entity_list.append((kb_id, entity, s, e))
    texts.append(text)

    # 存在实体才加入
    if len(entity_list) > 0:
        train_data.append({
            'text_id': text_id,
            'text': text,
            'entity_start': entity_start,
            'entity_end': entity_end,
            'entity_list': entity_list
        })
    else:
        print(idx)
f.close()

# 读取知识库
f = open('./data/kb_data', 'r', encoding='utf-8')
kb_data = []
entity_type = []
for i in enumerate(f):
    kb_data_line = json.loads(i[1])
    kb_data.append(kb_data_line)
    entity_type += kb_data_line['type']

type_set = sorted(list(set(entity_type)))
type_idx = {entity_type: idx for idx, entity_type in enumerate(type_set)}
idx_type = {idx: entity_type for idx, entity_type in enumerate(type_set)}

# 合并实体信息data
kb_data_preprocess = []
for i in kb_data:
    data = ''
    for j in i['data']:
        info = j['object'].replace(' ', '')
        if info not in data:
            if info[-1] in ['。', '；']:
                data += info
            else:
                data += (info + '。')
    kb_data_one = {
        'alias': [j.lower() for j in i.get('alias', [])],
        'subject_id': i['subject_id'].lower(),
        'subject': i.get('subject', ''),
        'type': [type_idx[j] for j in i['type']],
        'data': data.lower()
    }
    kb_data_preprocess.append(kb_data_one)
    texts.append(data)

# 构建文字编码字典
if os.path.exists('./data_deal/tokenizer.pkl'):
    with open('./data_deal/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
else:
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    with open('./data_deal/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
print('num_words: ', len(tokenizer.word_index))

# 保存编码字典
if os.path.exists('./data_deal/%d/word_index.pkl' % num_words):
    with open('./data_deal/%d/word_index.pkl' % num_words, 'rb') as f:
        word_index = pickle.load(f)
else:
    total_num = len(tokenizer.word_index)
    word_index = {}
    for word in tokenizer.word_index:
        word_id = tokenizer.word_index[word]
        if word_id <= num_words:
            word_index.update({word: word_id})
    with open('./data_deal/%d/word_index.pkl' % num_words, 'wb') as f:
        pickle.dump(word_index, f)

# 训练集文字转编码
text_len = []
for i in train_data:
    text_len.append(len(i['text']))
    i.update({
        'text_seq': [word_index.get(c, num_words + 1) for c in i['text']]
    })
print('最大长度', max(text_len))  # 49

# 保存处理好的训练集
with open('./data_deal/%d/train_data.pkl' % num_words, 'wb') as f:
    pickle.dump(train_data, f)

# 构建实体词典，用于训练，根据"id"检索
subject_data = defaultdict(dict)
for i in kb_data_preprocess:
    data_seq = [word_index.get(c, num_words + 1) for c in i['data']]
    if len(data_seq) == 0:
        i['data'] = '无'
        data_seq = [0]
    subject_data.update({
        i['subject_id']: {
            'type': i['type'],
            'data': i['data'],
            'data_seq': data_seq,
            'alias': i['alias']
        }
    })

# 保存处理好的实体词典
with open('./data_deal/%d/subject_data.pkl' % num_words, 'wb') as f:
    pickle.dump(subject_data, f)

del subject_data

# 构建实体词典，用于推断，根据"实体-id"检索
alias_data = defaultdict(dict)
for i in kb_data_preprocess:
    data_seq = [word_index.get(c, num_words + 1) for c in i['data']]
    if len(data_seq) == 0:
        i['data'] = '无'
        data_seq = [0]
    alias_list = i['alias'] if len(i['subject']) == 0 else i['alias'] + [i['subject']]
    for j in set(alias_list):
        alias_data[j.lower()].update({
            i['subject_id']: {
                'type': i['type'],
                'data': i['data'],
                'data_seq': data_seq
            }
        })

# 保存处理好的实体词典
with open('./data_deal/%d/alias_data.pkl' % num_words, 'wb') as f:
    pickle.dump(alias_data, f)
