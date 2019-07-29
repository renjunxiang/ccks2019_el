import torch
import torch.nn as nn
from .Embeds import Embeds


class Features_Link(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=300,
                 embedding=None,
                 hidden_dim=512,
                 device='cpu'):
        super(Features_Link, self).__init__()
        self.device = device

        # 字向量
        self.word_embeds = Embeds(vocab_size,
                                  embedding_dim,
                                  embedding,
                                  device)
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, sentence_seqs):
        # 词嵌入
        embedding = self.word_embeds(sentence_seqs)

        # 特征抽取
        embedding_t = embedding.transpose(2, 1)
        sentence_features = self.conv(embedding_t)
        sentence_features = sentence_features.transpose(2, 1)

        return sentence_features
