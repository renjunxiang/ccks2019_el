import torch
import torch.nn as nn


class Locate_Entity(nn.Module):
    """
    输入 sentence，预测 entity的首尾位置
    """

    def __init__(self,
                 hidden_dim=256,
                 device='cpu'):
        super(Locate_Entity, self).__init__()

        self.predict_B = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        self.predict_E = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        self.device = device

    def cal_loss(self,
                 s_B_idxs,
                 s_E_idxs,
                 s_B_labels,
                 s_E_labels,
                 mask_idx,
                 weight=4, ):
        mask_idx = mask_idx.float()

        # 计算subject_B的损失,提高正样本权重,去除mask部分
        loss1 = nn.BCEWithLogitsLoss(reduce=False)(s_B_idxs, s_B_labels)
        weight1 = torch.where(s_B_labels == 1, s_B_labels + weight - 1., s_B_labels + 1.)
        loss1 = loss1 * weight1
        loss1 = (loss1 * mask_idx).sum() / mask_idx.sum()

        # 计算subject_E的损失,提高正样本权重,去除mask部分
        loss2 = nn.BCEWithLogitsLoss(reduce=False)(s_E_idxs, s_E_labels)
        weight2 = torch.where(s_E_labels == 1, s_E_labels + weight - 1., s_E_labels + 1.)
        loss2 = loss2 * weight2
        loss2 = (loss2 * mask_idx).sum() / mask_idx.sum()

        return loss1 + loss2

    def forward(self, sentence_features):
        s_B_scores = self.predict_B(sentence_features)
        s_E_scores = self.predict_E(sentence_features)

        return s_B_scores.squeeze(-1), s_E_scores.squeeze(-1)


def slice_entity(batch_input, batch_slice):
    """
    1.从batch_input做切片,取batch_slice作为索引
    2.计算均值作为entity的语义
    :param
    batch_input: [batch_size, time_step, hidden_dim]
    :param
    batch_slice: [batch_size, 2]
    :return:

    batch_input = torch.Tensor([
        [[1, 2], [2, 3], [3, 4]],
        [[2, 3], [3, 4], [4, 5]],
        [[3, 4], [4, 5], [5, 6]]
        ])
    batch_slice = torch.LongTensor([
        [0, 1],
        [1, 2],
        [0, 2]
        ])
    return = torch.Tensor([
        [1.5000, 2.5000],
        [3.5000, 4.5000],
        [4.0000, 5.0000]])
    """
    shape_input = batch_input.size()
    batch_slice = batch_slice.long().unsqueeze(2).repeat(1, 1, shape_input[2])
    entity_slice = torch.gather(batch_input, 1, batch_slice)
    entity_slice = entity_slice.mean(dim=1)

    return entity_slice

