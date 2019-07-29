import torch
import torch.nn as nn


class Embeds(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=256,
                 embedding=None,
                 device='cpu'):
        super(Embeds, self).__init__()
        self.device = device
        # 词嵌入
        self.vocab_size = vocab_size

        # 字向量
        if embedding is None:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        else:
            embedding_dim = embedding.shape[1]
            self.word_embeds = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.embedding_dim = embedding_dim

    def forward(self, sentence_seqs):
        # 词嵌入
        embedding = self.word_embeds(sentence_seqs)

        return embedding
