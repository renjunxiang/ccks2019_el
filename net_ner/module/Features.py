import torch
import torch.nn as nn


class Features(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=256,
                 device='cpu'):
        super(Features, self).__init__()
        self.device = device

        # 文本信息
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(768,
                            hidden_dim // 2,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            # nn.ReLU(),
            # nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            # nn.ReLU()
        )

    def forward(self, sentence_features, mask_loss):
        # 序列信息
        sentence_features, self.hidden = self.lstm(sentence_features)

        # 特征抽取
        sentence_features_ = sentence_features.transpose(2, 1)
        sentence_features_ = self.conv(sentence_features_)
        sentence_features_ = sentence_features_.transpose(2, 1)

        # 残差
        sentence_features = sentence_features + sentence_features_
        mask = torch.eq(mask_loss, 0).unsqueeze(2).repeat(1, 1, self.hidden_dim)
        sentence_features = sentence_features.masked_fill(mask, 0)

        return sentence_features, mask_loss
