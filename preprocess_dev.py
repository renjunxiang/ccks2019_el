import json
import re
import os
import pickle

"""
测试集
{
    'text_id': '90000',
    'text': '电影《尹灵芝》开拍',
    'text_seq': [38, 47, 9, 1757, 540, 1754, 10, 191, 739]
}

"""

num_words = 10000

with open('./data_deal/%d/word_index.pkl' % num_words, 'rb') as f:
    word_index = pickle.load(f)

# 处理测试集
text_len = []
f = open('./data/develop.json', 'r', encoding='utf-8')
develop_data = []
for idx, i in enumerate(f):
    data_one = json.loads(i)
    text_id = data_one['text_id']
    text = data_one['text']
    text_len.append(len(text))

    develop_data.append({
        'text_id': text_id,
        'text': text,
        'text_seq': [word_index.get(c, num_words + 1) for c in text.lower()]
    })
f.close()

print('最大长度', max(text_len))

# 保存处理好的测试集
with open('./data_deal/%d/develop_data.pkl' % num_words, 'wb') as f:
    pickle.dump(develop_data, f)
