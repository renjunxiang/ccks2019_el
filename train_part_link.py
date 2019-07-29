import pickle
import os
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from net import MyDataset, collate_fn, collate_fn_link, deal_eval, seqs2batch, dataset
from net import Net
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', default='0', help='cuda:0/1/2')
parser.add_argument('--pretrain', default='bert', help='bert,wwm,ernie')
parser.add_argument('--num_layers', default=4, type=int, help='lstm layernum 3/4')
parser.add_argument('--hidden_dim', default=768, type=int, help='lstm hidden 768/1024')
parser.add_argument('--loss_weight', default=3, type=int, help='loss:2/3')
parser.add_argument('--num_words', default=10000, type=int, help='num_words:9000/10000')
parser.add_argument('--max_len', default=400, type=int, help='max_len:300/400/500')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--ner_id', default=1, type=int, help='None')
parser.add_argument('--k', default=0.813, type=float, help='k')
parser.add_argument('--lr', default=0.001, type=float, help='k')
parser.add_argument('--n', default=1, type=int, help='n')

opt = parser.parse_args()

dataset.device = "cuda:%s" % opt.cuda
device = dataset.device
# torch.manual_seed(1)


EMBEDDING_DIM = 300
embedding_name = opt.pretrain
num_layers = opt.num_layers
hidden_dim = opt.hidden_dim
BS = 64
num_words = opt.num_words
max_len = opt.max_len
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
    dataset=MyDataset(train1_data, subject_data, alias_data, opt.n),
    batch_size=BS, shuffle=True, collate_fn=collate_fn_link)

k = opt.k
for num_words in [opt.num_words]:
    for max_len in [opt.max_len]:
        for embedding_name in [opt.pretrain]:  # ['bert','wwm','ernie']
            bert_path = './pretrain/' + embedding_name + '/'
            dataset.tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt')
            dataset.BERT = BertModel.from_pretrained(bert_path).to(device)
            dataset.BERT.eval()
            dataset.max_len = max_len
            for loss_weight in [opt.loss_weight]:
                accu_ = 0
                while accu_ < k:
                    # vocab_size还有pad和unknow，要+2
                    model = Net(vocab_size=len(word_index) + 2,
                                embedding_dim=EMBEDDING_DIM,
                                num_layers=num_layers,
                                hidden_dim=hidden_dim,
                                embedding=embedding,
                                device=device).to(device)

                    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

                    # ==========导入ner预训练结果==========
                    ner_model = 'lstm_%d_%d_%d' % (num_layers, hidden_dim, loss_weight)
                    checkpoint = torch.load('./results_ner/%s/%s/%03d.pth' % (
                        embedding_name, ner_model, opt.ner_id), map_location=device)

                    # 仅导入ner部分
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)

                    with open('./results_ner/%s/%s/train_2_%03d.pkl' % (
                            embedding_name, ner_model, opt.ner_id), 'rb') as f:
                        ner_train_2 = pickle.load(f)

                    with open('./results_ner/%s/%s/dev_%03d.pkl' % (
                            embedding_name, ner_model, opt.ner_id), 'rb') as f:
                        ner_dev = pickle.load(f)

                    # ===================================

                    file_name = 'lstm_%d_%d_%d_len_%d_lf_2_l_2' % (
                        num_layers, hidden_dim, loss_weight, max_len)

                    if not os.path.exists('./results/%d/%s/%s/' % (num_words, embedding_name, file_name)):
                        os.mkdir('./results/%d/%s/%s/' % (num_words, embedding_name, file_name))

                    score1 = []
                    for epoch in range(epochs):
                        print('Start Epoch: %d\n' % (epoch + 1))
                        sum_link_loss = 0.0
                        model.train()
                        for i, data in enumerate(trainloader1):
                            data_ner, data_link = data

                            # 训练link,由于link的数量远多于ner,所以单独抽出来
                            text_seqs_link, kb_seqs_link, labels_link = data_link
                            nums = (len(text_seqs_link) - 1) // BS + 1
                            for n in range(nums):
                                optimizer.zero_grad()
                                text_seqs, _ = seqs2batch(text_seqs_link[(n * BS):(n * BS + BS)])
                                text_seqs = torch.LongTensor(text_seqs).to(device)
                                kb_seqs, _ = seqs2batch(kb_seqs_link[(n * BS):(n * BS + BS)])
                                kb_seqs = torch.LongTensor(kb_seqs).to(device)
                                link_labels = torch.Tensor(labels_link[(n * BS):(n * BS + BS)]).to(device)

                                # link损失
                                link_loss = model.cal_link_loss(text_seqs,
                                                                kb_seqs,
                                                                link_labels)
                                link_loss.backward()
                                optimizer.step()
                                sum_link_loss += link_loss.item() / nums

                            if (i + 1) % 200 == 0:
                                print('\nEpoch: %d ,batch: %d' % (epoch + 1, i + 1))
                                print('link_loss: %f' % (sum_link_loss / 200))
                                sum_link_loss = 0.0

                        # train2得分=====================================================================
                        model.eval()
                        p_len = 0.001
                        l_len = 0.001
                        correct_len = 0.001
                        score_list = []
                        entity_list_all = []

                        p_len1 = 0.001
                        l_len1 = 0.001
                        correct_len1 = 0.001
                        score_list1 = []

                        for idx, data in enumerate(train2_data):
                            model.zero_grad()
                            text_seqs = deal_eval([data])
                            text_seqs = text_seqs.to(device)
                            text = data['text']
                            with torch.no_grad():
                                entity_predict = model(text_seqs,
                                                       text,
                                                       alias_data,
                                                       ner_train_2[idx])

                            entity_list_all.append(entity_predict)

                            p_set = set([j[:-1] for j in entity_predict])
                            p_len += len(p_set)
                            l_set = set(data['entity_list'])
                            l_len += len(l_set)
                            correct_len += len(p_set.intersection(l_set))

                            p_set1 = set([j[1:-1] for j in entity_predict])
                            p_len1 += len(p_set1)
                            l_set1 = set([j[1:] for j in data['entity_list']])
                            l_len1 += len(l_set1)
                            correct_len1 += len(p_set1.intersection(l_set1))

                            if (idx + 1) % 2000 == 0:
                                print('finish train_2 %d' % (idx + 1))

                        Precision = correct_len / p_len
                        Recall = correct_len / l_len
                        F1 = 2 * Precision * Recall / (Precision + Recall)

                        Precision1 = correct_len1 / p_len1
                        Recall1 = correct_len1 / l_len1
                        F1_1 = 2 * Precision1 * Recall1 / (Precision1 + Recall1)

                        accu = F1 / F1_1
                        score1.append([epoch + 1,
                                       round(Precision1, 4), round(Recall1, 4), round(F1_1, 4), round(accu, 4),
                                       round(Precision, 4), round(Recall, 4), round(F1, 4)])
                        print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (epoch + 1, Precision, Recall, F1))

                        score1_df = pd.DataFrame(score1,
                                                 columns=['Epoch',
                                                          'P_n', 'R_n', 'F_n', 'link_accu',
                                                          'P', 'R', 'F1'])
                        print(score1_df)
                        score1_df.to_csv('./results/%d/%s/%s/new_train_2.csv' % (num_words, embedding_name, file_name),
                                         index=False)

                        accu_ = max(accu_, accu)
                        if accu >= opt.k:
                            # 保存网络参数
                            torch.save(model.state_dict(),
                                       './results/%d/%s/%s/new_param_%03d.pth' % (
                                           num_words, embedding_name, file_name, epoch + 1))

                            torch.save(model,
                                       './results/%d/%s/%s/new_%03d.pth' % (
                                           num_words, embedding_name, file_name, epoch + 1))

                            with open('./results/%d/%s/%s/new_train_2_%03d.pkl' % (
                                    num_words, embedding_name, file_name, epoch + 1),
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
                                    entity_predict = model(text_seq,
                                                           text,
                                                           alias_data,
                                                           ner_dev[idx])

                                entity_list_all.append(entity_predict)

                                if (idx + 1) % 1000 == 0:
                                    print('finish dev %d' % (idx + 1))
                            with open('./results/%d/%s/%s/new_dev_%03d.pkl' % (
                                    num_words, embedding_name, file_name, epoch + 1),
                                      'wb') as f:
                                pickle.dump(entity_list_all, f)
