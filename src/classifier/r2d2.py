import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE
from classifier.contrastive_loss import Contrastive_Loss, Contrastive_Loss_base
from classifier.LG import SG
from classifier.my_loss import LG_loss


class R2D2(BASE):

    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)
        self.args = args
        self.ebd_dim = ebd_dim
        # 温度系数暂且为5
        self.contrast_loss = Contrastive_Loss(args)
        self.my_loss = LG_loss(args)
        self.sg = SG(args)

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
            XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

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
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''
        YS1 = torch.tensor(YS1, dtype=torch.long).to(self.args.device)
        YQ1 = torch.tensor(YQ1, dtype=torch.long).to(self.args.device)
        YS, YQ = self.reidx_y(YS1, YQ1)

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

        loss = 0

        if self.args.add_cos:
            loss += F.cross_entropy(pred, YQ)
            pred = torch.argmax(pred, dim=1)
        if self.args.add_pro:
            contrast_loss_pro = self.contrast_loss(
                torch.cat((YC, YC), 0), protolabel, protolabel)
            loss += contrast_loss_pro * self.args.alpha_pro
        if self.args.add_instance:
            contrast_loss_instance = self.contrast_loss(
                torch.cat((YS, YQ), 0), XS, XQ)
            loss += contrast_loss_instance * self.args.alpha_pro
        if self.args.add_prosq:
            loss += self.my_loss(torch.cat((YS, YQ), 0),
                                 torch.cat((XS, XQ), 0), torch.cat((LS, LQ), 0))
        if self.args.add_prol:
            loss += self.my_loss(YC, protolabel,
                                 protolabel) * self.args.alpha_pl

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)

        pred = (10.0 ** self.alpha) * XQ @ W + self.beta
        pred = torch.argmax(pred, dim=1)
        # import pdb; pdb.set_trace()

        # loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)

        return acc, loss
