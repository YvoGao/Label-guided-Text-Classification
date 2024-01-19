import os
import time
import datetime

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from termcolor import colored
from train.utils import named_grad_param, grad_param, get_norm
from dataset.sampler import FewshotSampler


def train(train_data, val_data, model, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
    # creating a tmp directory to save the models
    # out_dir = os.path.abspath(os.path.join(
    #     os.path.curdir,
    #     "tmp-runs",
    #     'tmp'))
    out_dir = os.path.abspath(os.path.join(
        os.path.curdir,
        "tmp-runs",
        str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    best_acc = 0
    sub_cycle = 0
    best_path = None

    opt = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    print("{}, Start training".format(datetime.datetime.now()), flush=True)

    train_gen = FewshotSampler(train_data, args, args.train_episodes)
    train_gen_val = FewshotSampler(train_data, args, args.val_episodes)
    val_gen = FewshotSampler(val_data, args, args.val_episodes, 'val')

    for ep in range(args.train_epochs):

        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'ebd': []}
        train_loss = []

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                                 ncols=80, leave=False, desc=colored('Training on train',
                                                                     'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            train_one(task, model, opt, args, grad, train_loss)

        # monitor train acc
        if ep % 5 == 0:
            acc, std = test(train_data, model, args, args.val_episodes, False,
                            train_gen_val.get_epoch(), 'train')
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}".format(
                datetime.datetime.now(),
                "ep", ep,
                "train",
                "acc:", acc, std,

            ), flush=True)
        # Evaluate validation accuracy
        cur_acc, cur_std = test(val_data, model, args, args.val_episodes, False,
                                val_gen.get_epoch(), state="val")

        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
               "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}, {:s}{:>7.4f} ").format(
            datetime.datetime.now(),
            "ep", ep,
            "val  ",
            "acc:", cur_acc, cur_std,
            "train stats",
            "ebd_grad:", np.mean(np.array(grad['ebd'])),
            "clf_grad:", np.mean(np.array(grad['clf'])),
            "train loss:", np.mean(np.array(train_loss))
        ), flush=True)

        # Update the current best model if val acc is better
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_path = os.path.join(out_dir, 'tmp')

            # save current model
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now(),
                best_path))

            torch.save(model['ebd'].state_dict(), best_path + '.ebd')
            torch.save(model['clf'].state_dict(), best_path + '.clf')
            print("cur_acc > best_acc: best_path:", best_path)

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

    print("{}, End of training. Restore the best weights".format(
        datetime.datetime.now()),
        flush=True)

    # restore the best saved model
    model['ebd'].load_state_dict(torch.load(best_path + '.ebd'))
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir,
            "saved-runs",
            str(args.way) + "-way_" + str(args.shot) + "-shot_" + args.dataset))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')
        print("in args.save: best_path:", best_path)

        print("{}, Save best model to {}".format(
            datetime.datetime.now(),
            best_path), flush=True)

        torch.save(model['ebd'].state_dict(), best_path + '.ebd')
        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def train_one(task, model, opt, args, grad, train_loss):
    '''
        Train the model on one sampled task.
    '''
    if args.classifier == 'newr2d2':
        model['ebd'].eval()
    else:
        model['ebd'].train()
    model['clf'].train()
    opt.zero_grad()
    # torch.cuda.empty_cache()  # 释放显存

    support, query = task

    XS, LS = model['ebd'](support)
    YS = [x.label for x in support]

    XQ, LQ = model['ebd'](query, True)
    YQ = [x.label for x in query]

    # Apply the classifier
    _, loss = model['clf'](XS, YS, XQ, YQ, LS, LQ, 'train')

    if loss is not None:
        # loss.requires_grad_(True)
        loss.backward()

    if torch.isnan(loss):
        # do not update the parameters if the gradient is nan
        print("NAN detected")
        # print(model['clf'].lam, model['clf'].alpha, model['clf'].beta)
        return

    if args.clip_grad is not None:
        nn.utils.clip_grad_value_(grad_param(model, ['ebd', 'clf']),
                                  args.clip_grad)

    grad['clf'].append(get_norm(model['clf']))
    grad['ebd'].append(get_norm(model['ebd']))
    train_loss.append(loss.item())

    opt.step()


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None, state="test"):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    # 如果不是只测试的话，需要注释这三行
    # if args.mode == 'test':
    #     best_path = os.path.abspath(os.path.join(
    #         os.path.curdir,
    #         "saved-runs/16845130915756900/best"))
    #     # best_path = 'save-runs/baseline/best'
    #     print(best_path)
    #     model['ebd'].load_state_dict(torch.load(best_path + '.ebd'))
    #     model['clf'].load_state_dict(torch.load(best_path + '.clf'))
    model['ebd'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        print("state: ", state)
        sampled_tasks = FewshotSampler(
            test_data, args, num_episodes, state=state).get_epoch()

    acc = []
    acc_knn = []
    acc_proen = []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    for task in sampled_tasks:
        acc.append(test_one(task, model, args, state))

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
            datetime.datetime.now(),
            colored("acc mean", "blue"),
            np.mean(acc),
            colored("std", "blue"),
            np.std(acc),
        ), flush=True)
    acc = np.array(acc)
    return np.mean(acc), np.std(acc)


def test_one(task, model, args, state='test'):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    # torch.cuda.empty_cache()  # 释放显存
    support, query = task

    XS, LS = model['ebd'](support)
    YS = [x.label for x in support]

    XQ, LQ = model['ebd'](query, True)
    YQ = [x.label for x in query]

    # Apply the classifier
    acc, _ = model['clf'](XS, YS, XQ, YQ, LS, LQ, state)
    return acc
