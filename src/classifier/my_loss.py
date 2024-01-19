import torch
import torch.nn as nn
import numpy as np


def dot_similarity(XS, XQ):
    # 相似度没必要进行标准化
    # dot = torch.matmul(
    #     XS.unsqueeze(0).unsqueeze(-2),
    #     XQ.unsqueeze(1).unsqueeze(-1)
    # )
    # dot = dot.squeeze(-1).squeeze(-1)

    # scale = (torch.norm(XS, dim=1).unsqueeze(0) *
    #          torch.norm(XQ, dim=1).unsqueeze(1))

    # scale = torch.max(scale,
    #                   torch.ones_like(scale) * 1e-8)

    # dist = 1 - dot/scale
    # return dist
    return torch.matmul(XS, XQ.t())


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


# class LG_loss(nn.Module):

#     def __init__(self, args):
#         super(LG_loss, self).__init__()
#         self.tau = args.T
#         self.args = args

#     def similarity(self, x1, x2):
#         # # Gaussian Kernel
#         if self.args.sim == 'l2':
#             M = euclidean_dist(x1, x2)
#             s = torch.exp(-M/self.tau)
#         else:
#             # dot product
#             M = dot_similarity(x1, x2)/self.tau
#             # s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
#             s = torch.exp(M)
#         return s

#     """
#     x: batch_size * 768
#     L: batch_size * 768
#     """


#     def forward(self, x, L):

#         # # 分子
#         # fenzi = 0
#         # for i in range(len(x)):
#         #     fenzi += self.similarity(x[i], L[i])
#         # # 分母
#         # fenmu = 0
#         # for i in x:
#         #     for j in L:
#         #         fenmu += self.similarity(i, j)

#         # loss = fenzi/fenmu
#         # import pdb
#         # pdb.set_trace()
#         s2cl = torch.sum(torch.stack([self.similarity(x_i, L_i) for x_i, L_i in zip(x, L)]))
#         s2alll = torch.sum(torch.stack([self.similarity(x_i, L_j) for x_i in x for L_j in L]))

#         loss = s2cl / s2alll

#         return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.pairwise_distance(anchor, positive)
        distance_negative = torch.pairwise_distance(anchor, negative)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()


class LG_loss(nn.Module):

    def __init__(self, args):
        super(LG_loss, self).__init__()
        self.tau = args.T
        self.args = args

    def similarity(self, x1, x2):
        # # Gaussian Kernel
        if self.args.sim == 'l2':
            M = euclidean_dist(x1, x2)
            s = torch.exp(-M/self.tau)
        else:
            # dot product
            M = dot_similarity(x1, x2)/self.tau
            s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_labels, X, L):

        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, L)

        # import pdb
        # pdb.set_trace()

        # 全一
        mask_i = torch.from_numpy(
            np.ones((len_, len_))).to(batch_labels.device)

        # # 对角线为0，其他为1
        # mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device)
        # 将（len,)-->(1, len) --> (len, len)
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        # 得到相同标签的位置
        mask_j = (batch_labels.unsqueeze(1) - label_matrix ==
                  0).float() * mask_i  # sum over items in the denominator

        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10)
        s_j = torch.clamp(s*mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i) * mask_j, 1)/pos_num
        loss = torch.mean(log_p)

        return loss
