import torch
from torch.utils.data import Dataset
from random import choice
from pytorch_pretrained_bert import BertTokenizer, BertModel

max_len = 400
device = "cuda:0"

tokenizer = None
BERT = None


def text2bert(texts):
    texts = [text.lower() for text in texts]
    mask_loss = []
    text_seqs = []
    segments_ids = []

    text_len = [min(max_len, 1 + len(text) + 1) for text in texts]
    text_max = max(text_len)

    for num, text in enumerate(texts):
        text_cat = ['[CLS]'] + list(text[:(max_len - 2)]) + ['[SEP]']
        text_bert = []
        for c in text_cat:
            if c in tokenizer.vocab:
                text_bert.append(c)
            else:
                text_bert.append('[UNK]')

        # 用于损失的mask，除了sentence其余都是0
        mask_loss.append([0] + [1] * (len(text_cat) - 2) + [0] * (text_max - text_len[num] + 1))

        # 输入bert
        text_seq = tokenizer.convert_tokens_to_ids(text_bert) + [0] * (text_max - text_len[num])
        text_seqs.append(text_seq)
        segments_ids.append([0] * text_max)
    text_seqs = torch.LongTensor(text_seqs).to(device)
    segments_ids = torch.LongTensor(segments_ids).to(device)

    # bert的mask编码
    mask_bert = 1 - torch.eq(text_seqs, 0)
    with torch.no_grad():
        sentence_features, _ = BERT(text_seqs, segments_ids, mask_bert)
    sentence_features = sentence_features[-1]

    mask_loss = torch.LongTensor(mask_loss).to(device)
    mask_feature = mask_loss.unsqueeze(-1).repeat(1, 1, 768)

    # 最终只保留sentence的序列输出
    sentence_features = torch.where(torch.eq(mask_feature, 0),
                                    torch.zeros_like(sentence_features),
                                    sentence_features)

    return sentence_features, mask_loss


# 定义数据读取方式
class MyDataset(Dataset):
    def __init__(self, dataset, subject_data, alias_data):
        self.dataset = dataset
        self.subject_data = subject_data
        self.alias_data = alias_data
        self.kb_ids = list(subject_data.keys())

    def __getitem__(self, index):
        data_one = self.dataset[index]
        entity_list = data_one['entity_list']
        entity_ses = []
        kb_seqs = []
        labels = []
        for entity_info in entity_list:
            kb_id, entity, s, e = entity_info

            kb_seq = self.subject_data[kb_id]['data_seq'][:max_len]
            kb_seqs.append(kb_seq)
            labels.append(1)
            # link_label == 0,当存在歧义，从歧义选0，否则任意抽一个0
            kb_id_other = choice(self.kb_ids)  # 抽样的效率
            while kb_id_other == kb_id:
                kb_id_other = choice(self.kb_ids)

            if entity in self.alias_data:
                if len(self.alias_data[entity]) > 1:
                    kb_id_others = list(self.alias_data[entity].keys())
                    if kb_id in kb_id_others:
                        kb_id_others.remove(kb_id)
                    kb_id_other = choice(kb_id_others)

            kb_seq = self.subject_data[kb_id_other]['data_seq'][:max_len]
            kb_seqs.append(kb_seq)
            labels.append(0)

        return data_one, kb_seqs, labels, entity_ses

    def __len__(self):
        return len(self.dataset)


def seqs2batch(seqs):
    seqs_len = [min(max_len, len(i)) for i in seqs]
    seqs_max = max(seqs_len)

    seqs_batch = []
    for num, seq in enumerate(seqs):
        seqs_batch.append(seq[:max_len] + [0] * (seqs_max - seqs_len[num]))

    return seqs_batch, seqs_max


def collate_fn(batch):
    # 实体识别
    text_seqs_ner, text_max = seqs2batch([i[0]['text_seq'] for i in batch])

    entity_starts = []
    entity_ends = []

    text_seqs_link = []
    kb_seqs_link = []
    labels_link = []

    for num, i in enumerate(batch):
        data_one, kb_seqs, labels, entity_ses = i

        entity_start = [0] * (text_max + 2)
        for j in data_one['entity_start']:
            entity_start[j + 1] = 1
        entity_starts.append(entity_start)

        entity_end = [0] * (text_max + 2)
        for j in data_one['entity_end']:
            entity_end[j + 1] = 1
        entity_ends.append(entity_end)

        text_seqs_link += [text_seqs_ner[num]] * len(labels)
        kb_seqs_link += kb_seqs
        labels_link += labels

    texts = [i[0]['text'] for i in batch]
    text_features, mask_loss_texts = text2bert(texts)
    entity_starts = torch.Tensor(entity_starts).to(device)
    entity_ends = torch.Tensor(entity_ends).to(device)

    return (text_features, mask_loss_texts, entity_starts, entity_ends), (text_seqs_link, kb_seqs_link, labels_link)


def deal_eval(batch):
    text_seq = []

    for i, j in enumerate(batch):
        text_seq.append(j['text_seq'])

    text_seq = torch.LongTensor(text_seq)

    return text_seq
