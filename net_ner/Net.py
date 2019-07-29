import torch
import torch.nn as nn
from .module import Features
from .task import Locate_Entity
from .dataset import text2bert


class Net(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=256,
                 num_layers=3,
                 hidden_dim=256,
                 embedding=None,
                 device='cpu'):
        super(Net, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        # 文本信息
        self.get_features_ner = Features(num_layers,
                                         hidden_dim,
                                         device)


        # entity起止位置，只做二分类
        self.get_entity_score = Locate_Entity(hidden_dim,
                                              device)


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

    def forward(self, text_seq, text, alias_data):
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

        return entity_predict
