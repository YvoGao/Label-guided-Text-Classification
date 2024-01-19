import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE
from classifier.contrastive_loss import Contrastive_Loss
from classifier.my_loss import LG_loss
from classifier.LG import SG


def dot_similarity(x1, x2):
    return torch.matmul(x1, x2.t())


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


class MBC(BASE):
    '''
        Metric-based Classifier FOR FEW SHOT LEARNING
    '''

    def __init__(self, ebd_dim, args):
        super(MBC, self).__init__(args)
        self.args = args
        self.ebd_dim = ebd_dim
        # 温度系数暂且为5
        self.contrast_loss = Contrastive_Loss(args)
        self.my_loss = LG_loss(args)
        self.sg = SG(args)

    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices]

        prototype = []
        classes = []
        for i in range(self.args.way):
            prototype.append(torch.mean(
                sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                keepdim=True))
            classes.append(sorted_YS[i*self.args.shot])
        prototype = torch.cat(prototype, dim=0)
        classes = torch.tensor(classes)
        return prototype, classes

    def forward(self, XS, YS1, XQ, YQ1, LS, LQ, state):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS1 (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ1 (support y): query_size
            @param LS (support x): support_size x ebd_dim
            @param LQ (support x): query_size x ebd_dim

            @return acc
            @return loss
        '''
        loss = 0
        YS1 = torch.tensor(YS1, dtype=torch.long).to(self.args.device)
        YQ1 = torch.tensor(YQ1, dtype=torch.long).to(self.args.device)
        YS, YQ = self.reidx_y(YS1, YQ1)

        # 不引导前计算损失

        XS = self.sg(XS, LS)
        prototypesentence, YC = self._compute_prototype(XS, YS)
        protolabel, YC = self._compute_prototype(LS, YS)
        if self.args.protype == "mean":
            prototype = (prototypesentence + protolabel)/2
        elif self.args.protype == "single":
            prototype = prototypesentence
        else:
            prototype = protolabel
        YC = YC.to(self.args.device)

        if self.args.cltype == 'proto':
            if self.args.sim == "l2":
                pred = -self._compute_l2(prototype, XQ)
            elif self.args.sim == "cos":
                pred = -self._compute_cos(prototype, XQ)
            if not self.args.add_cos:
                pred = torch.argmax(pred, dim=1)
        elif self.args.cltype == 'knn':
            if self.args.sim == "l2":
                pred = -self._compute_l2(torch.cat((XS, protolabel), 0), XQ)
            elif self.args.sim == "cos":
                pred = -self._compute_cos(torch.cat((XS, protolabel), 0), XQ)
            pred = torch.argmax(pred, dim=1)
            YS = torch.cat((YS, YC), 0)
            pred = YS[pred]
        elif self.args.cltype == 'label':
            if self.args.sim == "l2":
                pred = -self._compute_l2(protolabel, XQ)

            elif self.args.sim == "cos":
                pred = -self._compute_cos(protolabel, XQ)
            if not self.args.add_cos:
                pred = torch.argmax(pred, dim=1)
        else:
            if self.args.sim == "l2":
                pred = - \
                    self._compute_l2(
                        torch.cat((prototypesentence, prototype, protolabel), 0), XQ)

            elif self.args.sim == "cos":
                pred = - \
                    self._compute_cos(
                        torch.cat((prototypesentence, prototype, protolabel), 0), XQ)
            pred = torch.argmax(pred, dim=1)
            YS = torch.cat((YC, YC, YC), 0)
            pred = YS[pred]

        if self.args.add_cos:
            loss += F.cross_entropy(pred, YQ)
            pred = torch.argmax(pred, dim=1)
        if self.args.add_pro:
            contrast_loss_pro = self.contrast_loss(
                torch.cat((YC, YC), 0), protolabel, protolabel)
            loss += contrast_loss_pro * self.args.alpha_pro
        if self.args.add_instance:
            # baseline
            # SG选用single
            contrast_loss_instance = self.contrast_loss(
                torch.cat((YS, YQ), 0), XS, XQ)
            loss += self.contrast_loss(
                torch.cat((YC, YC), 0), prototypesentence, prototypesentence)
            loss += contrast_loss_instance * self.args.alpha_pro

        if self.args.add_prosq:
            # loss += self.my_loss(YQ, XQ, LQ)
            # loss += self.my_loss(YS, XS, LS)
            loss += self.my_loss(torch.cat((YS, YQ), 0),
                                 torch.cat((XS, XQ), 0), torch.cat((LS, LQ), 0))
        if self.args.add_prol:
            loss += self.my_loss(YC, protolabel,
                                 protolabel) * self.args.alpha_pl

        acc = BASE.compute_acc(pred, YQ)

        return acc, loss

    def _compute_l2(self, XS, XQ):
        '''
            Compute the pairwise l2 distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size x support_size

        '''

        diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        return dist
