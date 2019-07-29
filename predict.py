import pickle
import os
import torch
from net import deal_eval, dataset
from pytorch_pretrained_bert import BertTokenizer, BertModel

device = 'cuda:0'
dataset.device = device
num_words = 10000
max_len = 400

# 读取实体词典，用于推断，根据"实体-id"检索
with open('./data_deal/%d/alias_data.pkl' % num_words, 'rb') as f:
    alias_data = pickle.load(f)

# 读取测试集预处理
with open('./data_deal/%d/test_data.pkl' % num_words, 'rb') as f:
    develop_data = pickle.load(f)

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
                # 11,  # 0.742 xin x
                12,  # 0.742 jiu
            ],
            'lstm_4_1024_2_len_400_lf_2_l_2': [
                21,  # 0.742 xin x
            ],
        },
    }
}

for num_words, value1 in file_name.items():
    for embedding_name, value2 in value1.items():
        bert_path = './data/pretrain/' + embedding_name + '/'
        dataset.tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt')
        dataset.BERT = BertModel.from_pretrained(bert_path).to(device)
        dataset.BERT.eval()
        dataset.max_len = max_len
        for model_name, model_idxs in value2.items():
            if not os.path.exists('./results_test/%d/%s/%s/' % (num_words, embedding_name, model_name)):
                os.mkdir('./results_test/%d/%s/%s/' % (num_words, embedding_name, model_name))
            for model_idx in model_idxs:
                model = torch.load('./results/%d/%s/%s/%03d.pth' % (
                    num_words, embedding_name, model_name, model_idx), map_location=device)
                model.device = device
                model.to(device)

                model.eval()
                entity_list_all = []
                for idx, data in enumerate(develop_data):
                    model.zero_grad()
                    text_seq = deal_eval([data])
                    text_seq = text_seq.to(device)
                    text = data['text']
                    with torch.no_grad():
                        entity_predict = model(text_seq,
                                               text,
                                               alias_data)

                    entity_list_all.append(entity_predict)

                with open('./results_test/%d/%s/%s/test_%03d.pkl' % (
                        num_words, embedding_name, model_name, model_idx),
                          'wb') as f:
                    pickle.dump(entity_list_all, f)
