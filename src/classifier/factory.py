import torch
from classifier.r2d2 import R2D2
from classifier.mbc import MBC
from dataset.utils import tprint


def get_classifier(ebd_dim, args):
    tprint("Building classifier")
    # import pdb
    # pdb.set_trace()

    if args.classifier == 'r2d2':
        model = R2D2(ebd_dim, args)
    elif args.classifier == 'mbc':
        model = MBC(ebd_dim, args)
    else:
        raise ValueError('Invalid classifier.'
                         'classifier can only be: mbc, r2d2.')
    if args.snapshot != '':
        # load pretrained models
        tprint("Loading pretrained classifier from {}".format(
            args.snapshot + '.clf'
        ))
        model.load_state_dict(torch.load(args.snapshot + '.clf'))

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
