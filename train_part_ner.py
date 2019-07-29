import pickle
import os
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from net_ner import MyDataset, collate_fn, deal_eval, seqs2batch, dataset
from net_ner import Net
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', default='0', help='cuda:0/1/2')
parser.add_argument('--pretrain', default='bert', help='bert,wwm,ernie')
parser.add_argument('--num_layers', default=4, type=int, help='lstm layernum 3/4')
parser.add_argument('--hidden_dim', default=768, type=int, help='lstm hidden 768/1024')
parser.add_argument('--loss_weight', default=3, type=int, help='loss:2/3')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--k', default=0.813, type=float, help='k')

opt = parser.parse_args()

dataset.device = "cuda:%s" % opt.cuda
device = dataset.device
# torch.manual_seed(1)


EMBEDDING_DIM = 300
num_layers = opt.num_layers
hidden_dim = opt.hidden_dim
BS = 64
num_words = 10000
epochs = opt.epochs

with open('./data_deal/%d/weight_baidubaike.pkl' % num_words, 'rb') as f:
    embedding = pickle.load(f)
    embedding = torch.FloatTensor(embedding).to(device)

# 导入文本编码、词典
with open('./data_deal/%d/word_index.pkl' % num_words, 'rb') as f:
    word_index = pickle.load(f)

# 读取训练集预处理
with open('./data_deal/%d/train_data.pkl' % num_words, 'rb') as f:
    train_data = pickle.load(f)

# 读取实体词典，用于训练，根据"id"检索
with open('./data_deal/%d/subject_data.pkl' % num_words, 'rb') as f:
    subject_data = pickle.load(f)

# 读取实体词典，用于推断，根据"实体-id"检索
with open('./data_deal/%d/alias_data.pkl' % num_words, 'rb') as f:
    alias_data = pickle.load(f)

# 读取验证集预处理
with open('./data_deal/%d/develop_data.pkl' % num_words, 'rb') as f:
    develop_data = pickle.load(f)

# 拆分训练集
train1_data, train2_data = train_test_split(train_data,
                                            test_size=0.1,
                                            random_state=1)
trainloader1 = torch.utils.data.DataLoader(
    dataset=MyDataset(train1_data, subject_data, alias_data),
    batch_size=BS, shuffle=True, collate_fn=collate_fn)

train2_data = train2_data
trainloader2 = torch.utils.data.DataLoader(
    dataset=MyDataset(train2_data, subject_data, alias_data),
    batch_size=BS, shuffle=False, collate_fn=collate_fn)

k = opt.k
pwd = '.'
for embedding_name in [opt.pretrain]:  # ['bert','wwm','ernie']
    bert_path = './pretrain/' + embedding_name + '/'
    dataset.tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt')
    dataset.BERT = BertModel.from_pretrained(bert_path).to(device)
    dataset.BERT.eval()
    dataset.max_len = 300
    for loss_weight in [opt.loss_weight]:
        F1_ = 0
        while F1_ < k:
            # vocab_size还有pad和unknow，要+2
            model = Net(vocab_size=len(word_index) + 2,
                        embedding_dim=EMBEDDING_DIM,
                        num_layers=num_layers,
                        hidden_dim=hidden_dim,
                        embedding=embedding,
                        device=device).to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            file_name = 'lstm_%d_%d_%d' % (
                num_layers, hidden_dim, loss_weight)

            if not os.path.exists('./results_ner/%s/%s/' % (embedding_name, file_name)):
                os.mkdir('./results_ner/%s/%s/' % (embedding_name, file_name))

            score1 = []
            for epoch in range(epochs):
                print('Start Epoch: %d\n' % (epoch + 1))
                sum_ner_loss = 0.0
                model.train()
                for i, data in enumerate(trainloader1):
                    data_ner, data_link = data

                    # 训练ner
                    model.zero_grad()
                    text_features, mask_loss_texts, entity_starts, entity_ends = data_ner

                    # ner损失
                    ner_loss = model.cal_ner_loss(text_features,
                                                  mask_loss_texts,
                                                  entity_starts,
                                                  entity_ends,
                                                  loss_weight)
                    ner_loss.backward()
                    optimizer.step()
                    sum_ner_loss += ner_loss.item()

                    if (i + 1) % 200 == 0:
                        print('\nEpoch: %d ,batch: %d' % (epoch + 1, i + 1))
                        print('ner_loss: %f' % (sum_ner_loss / 200))
                        sum_ner_loss = 0.0

                # train2得分=====================================================================
                model.eval()
                p_len = 0.001
                l_len = 0.001
                correct_len = 0.001
                score_list = []
                entity_list_all = []

                for idx, data in enumerate(train2_data):
                    model.zero_grad()
                    text_seqs = deal_eval([data])
                    text_seqs = text_seqs.to(device)
                    text = data['text']
                    with torch.no_grad():
                        entity_predict = model(text_seqs, text, alias_data)

                    entity_list_all.append(entity_predict)

                    p_set = set(entity_predict)
                    p_len += len(p_set)
                    l_set = set([j[1:] for j in data['entity_list']])
                    l_len += len(l_set)
                    correct_len += len(p_set.intersection(l_set))

                    if (idx + 1) % 2000 == 0:
                        print('finish train_2 %d' % (idx + 1))

                Precision = correct_len / p_len
                Recall = correct_len / l_len
                F1 = 2 * Precision * Recall / (Precision + Recall)

                score1.append([epoch + 1,
                               round(Precision, 4), round(Recall, 4), round(F1, 4)])
                print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (epoch + 1, Precision, Recall, F1))

                score1_df = pd.DataFrame(score1,
                                         columns=['Epoch',
                                                  'P', 'R', 'F1'])
                print(score1_df)
                score1_df.to_csv('./results_ner/%s/%s/new_train_2.csv' % (embedding_name, file_name),
                                 index=False)
                F1_ = max(F1_, F1)
                if F1 >= k:
                    # 保存网络参数
                    torch.save(model.state_dict(),
                               pwd + '/results_ner/%s/%s/new_%03d.pth' % (
                                   embedding_name, file_name, epoch + 1))
                    with open('./results_ner/%s/%s/new_train_2_%03d.pkl' % (embedding_name, file_name, epoch + 1),
                              'wb') as f:
                        pickle.dump(entity_list_all, f)

                    # eval预测结果=====================================================================

                    model.eval()
                    entity_list_all = []
                    for idx, data in enumerate(develop_data):
                        model.zero_grad()
                        text_seq = deal_eval([data])
                        text_seq = text_seq.to(device)
                        text = data['text']
                        with torch.no_grad():
                            entity_predict = model(text_seq, text, alias_data)

                        entity_list_all.append(entity_predict)

                        if (idx + 1) % 1000 == 0:
                            print('finish dev %d' % (idx + 1))
                    with open('./results_ner/%s/%s/new_dev_%03d.pkl' % (embedding_name, file_name, epoch + 1),
                              'wb') as f:
                        pickle.dump(entity_list_all, f)
