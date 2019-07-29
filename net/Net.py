import torch
import torch.nn as nn
from .module import Features, Features_Link
from .task import Locate_Entity, Link_KB
from .dataset import text2bert, seqs2batch


class Net(nn.Module):
    """
    1.输入text，预测entity的位置
    2.输入entity和知识库，预测最大概率的链接
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim=256,
                 num_layers=3,
                 hidden_dim=256,
                 embedding=None,
                 device='cpu'):
        super(Net, self).__init__()

        self.slice = slice
        self.device = device
        self.hidden_dim = hidden_dim

        # 文本信息
        self.get_features_ner = Features(num_layers,
                                         hidden_dim,
                                         device)

        self.get_features_link = Features_Link(vocab_size,
                                               embedding_dim,
                                               embedding,
                                               hidden_dim,
                                               device)

        # entity起止位置，只做二分类
        self.get_entity_score = Locate_Entity(hidden_dim,
                                              device)

        # 实体链接得分
        self.link_kb = Link_KB(hidden_dim, device)

    def cal_ner_loss(self,
                     text_features,
                     mask_loss_texts,
                     entity_B_labels,
                     entity_E_labels,
                     loss_weight):
        # 计算实体文本语义
        text_features, mask_loss_texts = self.get_features_ner(text_features, mask_loss_texts)

        # 预测entity的起止
        entity_B_scores, entity_E_scores = self.get_entity_score(text_features)
        mask_loss_texts = mask_loss_texts.float()

        # 计算entity的损失,去除mask部分
        loss = self.get_entity_score.cal_loss(entity_B_scores,
                                              entity_E_scores,
                                              entity_B_labels,
                                              entity_E_labels,
                                              mask_loss_texts,
                                              loss_weight)

        return loss

    def cal_link_loss(self,
                      text_seqs,
                      kb_seqs,
                      link_labels
                      ):
        # 计算做链接的语义
        entity_features = self.get_features_link(text_seqs)

        # 计算kb文本语义
        kb_features = self.get_features_link(kb_seqs)

        # 计算链接得分的损失
        loss = self.link_kb.cal_loss(entity_features,
                                     kb_features,
                                     link_labels)

        return loss

    def forward(self, text_seq, text, alias_data, entity_predict=None):
        if entity_predict is None:
            entity_features, mask_loss = text2bert([text])
            entity_features, _ = self.get_features_ner(entity_features, mask_loss)

            # 预测实体
            entity_predict = []
            entity_B_scores, entity_E_scores = self.get_entity_score(entity_features)
            entity_B_scores = entity_B_scores[:, 1:]
            entity_E_scores = entity_E_scores[:, 1:]

            entity_B_scores = nn.Sigmoid()(entity_B_scores[0]).tolist()
            entity_E_scores = nn.Sigmoid()(entity_E_scores[0]).tolist()
            for entity_B_idx, entity_B_score in enumerate(entity_B_scores):
                if entity_B_score > 0.5:
                    # E是在B之后的,索引从B开始
                    for entity_E_idx, entity_E_score in enumerate(entity_E_scores[entity_B_idx:]):
                        if entity_E_score > 0.5:
                            entity_idx = [entity_B_idx, entity_B_idx + entity_E_idx]

                            entity = text[entity_idx[0]:(entity_idx[1] + 1)]
                            if entity in alias_data:
                                entity_predict.append(
                                    (text[entity_idx[0]:(entity_idx[1] + 1)], entity_idx[0], entity_idx[1]))
                            break

        # 链接知识库
        results = []
        for entity_info in entity_predict:
            entity, s, e = entity_info

            # 计算做链接的语义
            entity_features = self.get_features_link(text_seq)

            if entity in alias_data:
                alias_predict = alias_data[entity.lower()]
                kb_ids = list(alias_predict.keys())

                num_batch = (len(kb_ids) - 1) // 64 + 1
                link_scores = []
                for n in range(num_batch):
                    kb_ids_batch = kb_ids[(n * 64):(n * 64 + 64)]
                    kb_seqs = [alias_predict[kb_id]['data_seq'] for kb_id in kb_ids_batch]
                    kb_seqs, kb_max = seqs2batch(kb_seqs)
                    kb_seqs = torch.LongTensor(kb_seqs).to(self.device)

                    # 计算kb文本语义
                    kb_features = self.get_features_link(kb_seqs)

                    # 计算batch链接得分
                    text_features_batch = entity_features.repeat(len(kb_ids_batch), 1, 1)
                    link_scores_batch = nn.Sigmoid()(self.link_kb(text_features_batch, kb_features)).tolist()
                    link_scores += link_scores_batch
                score = max(link_scores)
                results.append((kb_ids[link_scores.index(score)], entity, s, e, score))

        return results
