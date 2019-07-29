import torch
import torch.nn as nn


class Link_KB(nn.Module):
    """
    输入 实体特征、知识特征，预测两者链接得分
    """

    def __init__(self,
                 hidden_dim,
                 device='cpu'):
        super(Link_KB, self).__init__()

        # 实体和知识库拼接，只做二分类
        self.cal_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.hidden_dim = hidden_dim
        self.device = device

    def cal_loss(self,
                 entity_features,
                 kb_features,
                 labels):
        entity_features_ = nn.MaxPool1d(entity_features.size()[1])(entity_features.transpose(2, 1)).squeeze(-1)
        kb_features_ = nn.MaxPool1d(kb_features.size()[1])(kb_features.transpose(2, 1)).squeeze(-1)

        features = entity_features_ * kb_features_
        features = torch.cat([entity_features_, kb_features_, features], -1)

        scores = self.cal_score(features).squeeze(-1)

        # 计算实体和知识库拼接得分的损失
        loss = nn.BCEWithLogitsLoss()(scores, labels)

        return loss

    def forward(self,
                entity_features,
                kb_features):
        entity_features_ = nn.MaxPool1d(entity_features.size()[1])(entity_features.transpose(2, 1)).squeeze(-1)
        kb_features_ = nn.MaxPool1d(kb_features.size()[1])(kb_features.transpose(2, 1)).squeeze(-1)

        features = entity_features_ * kb_features_
        features = torch.cat([entity_features_, kb_features_, features], -1)

        scores = self.cal_score(features).squeeze(-1)

        return scores
