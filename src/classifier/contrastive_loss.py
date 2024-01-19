import torch
import torch.nn as nn
import numpy as np


class Contrastive_Loss_base(nn.Module):

    def __init__(self, tau=5.0):
        super(Contrastive_Loss_base, self).__init__()
        self.tau = tau

    def similarity(self, x1, x2):
        # # Gaussian Kernel

        # M = euclidean_dist(x1, x2)
        # s = torch.exp(-M/self.tau)

        # dot product
        M = dot_similarity(x1, x2)/self.tau
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_label, *x):
        X = torch.cat(x, 0)
        batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        if len(x) == 1:
            mask_i = torch.from_numpy(
                np.ones((len_, len_))).to(batch_labels.device)
        else:
            # sum over items in the numerator
            mask_i = 1. - \
                torch.from_numpy(np.identity(len_)).to(batch_labels.device)
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix ==
                  0).float()*mask_i  # sum over items in the denominator
        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10)
        s_j = torch.clamp(s*mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        loss = torch.mean(log_p)

        return loss


def dot_similarity(XS, XQ):
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

class Contrastive_Loss(nn.Module):

    def __init__(self, args):
        super(Contrastive_Loss, self).__init__()
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

    def forward(self, batch_labels, *x):
        X = torch.cat(x, 0)
        # batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        len_ = batch_labels.size()[0]
        # import pdb
        # pdb.set_trace()
        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        if len(x) == 1:
            mask_i = torch.from_numpy(
                np.ones((len_, len_))).to(batch_labels.device)
        else:
            # sum over items in the numerator
            mask_i = 1. - \
                torch.from_numpy(np.identity(len_)).to(batch_labels.device)
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix ==
                  0).float() * mask_i  # sum over items in the denominator
        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s * mask_i, 1), min=1e-10)
        s_j = torch.clamp(s * mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        loss = torch.mean(log_p)

        return loss
