import re
import argparse
import os
import shutil
import time
import math

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

import lp.db_semisuper as db_semisuper
from tqdm import tqdm


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value) for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    # LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    print("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print("--- checkpoint copied to %s ---" % best_path)


def create_data_loaders(train_transformation, eval_transformation, datadir, args):
    """create dataloaders with args and train/eval transformations
        args.
            exclude_unlabeled: only use labeled data, lower bound
            fully_supervised: use total data, upper bound
            labeled_batch_size: number of labeled samples in a batch, semi-supervised methods
                use `mean_teacher.data.TwoStreamBatchSampler()` to merge a batch samples from labeled and unlabeled data.

    Args:
        train_transformation ([type]): transforms.Compose([..])
        eval_transformation ([type]): transforms.Compose([..])
        datadir ([type]): dataset image dir
        args ([type]): parsed args

    Returns:
        train_loader: trainset dataloader
        eval_loader: evalset dataloader
        train_loader_noshuff: a noshuff dataloader using total data with twice batchsize.
        dataset: lp.db_semisuper.DBSS, which includes `p_labels`, `p_weights` and `class_weights`.
    """
    traindir = os.path.join(datadir, args.train_subdir)  # train+val
    evaldir = os.path.join(datadir, args.eval_subdir)  # test

    # can only set 1 mode from 3 candidate modes
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size, args.fully_supervised])

    # ref: torchvision.datasets.ImageFolder
    dataset = db_semisuper.DBSS(traindir, train_transformation)

    if not args.fully_supervised and args.labels:  # data-local/labels/cifar10/500_balanced_labels/10.txt
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())  # {img_name: cls_name}
            # labels is a dict, cuz `data.relabel_dataset()` has query demands
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)  # only labeled
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.fully_supervised:
        sampler = SubsetRandomSampler(range(len(dataset)))  # all data
        dataset.labeled_idx = range(len(dataset))
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:  # semi, two stream: label + unlabel
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs,
            labeled_idxs,
            args.batch_size,
            args.labeled_batch_size,
        )
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    # use different batch sampler
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=False)  # False cuz torch warnings

    train_loader_noshuff = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,  # Needs images twice as fast
        pin_memory=False,
        drop_last=False)

    eval_dataset = torchvision.datasets.ImageFolder(evaldir, eval_transformation)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=False,
                                              drop_last=False)

    return train_loader, eval_loader, train_loader_noshuff, dataset


def update_ema_variables(model, ema_model, alpha, global_step):
    # note Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def create_model(num_classes, args, ema=False):
    model_factory = architectures.__dict__[args.arch]
    model_params = dict(pretrained=args.pretrained,
                        num_classes=num_classes,
                        isL2=args.isL2,
                        double_output=args.double_output)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()  # require_grad=False
    return model


def train(train_loader, model, optimizer, epoch, global_step, args, ema_model=None):
    """[summary]

    Args:
        train_loader ([type]): [description]
        model ([type]): [description]
        optimizer ([type]): [description]
        epoch ([type]): [description]
        global_step ([type]): [description]
        args ([type]): [description]
        ema_model ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    class_criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='none').cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    isMT = ema_model is not None

    # switch to train mode
    model.train()
    if isMT:
        ema_model.train()

    end = time.time()

    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        if isMT:  # data.TransformTwice
            input = batch_input[0]
            ema_input = batch_input[1]  # ema input
        else:
            input = batch_input

        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)

        meters.update('lr', optimizer.param_groups[0]['lr'])

        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())
        weight_var = torch.autograd.Variable(weight.cuda())  # pseudo sample weight
        c_weight_var = torch.autograd.Variable(c_weight.cuda())  # class weight

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        if isMT:
            ema_input_var = torch.autograd.Variable(ema_input.cuda())
            ema_logit, _, _ = ema_model(ema_input_var)
            class_logit, cons_logit, _ = model(input_var)  # out two fc

            ema_logit = torch.autograd.Variable(ema_logit.detach().data, requires_grad=False)

            if args.logit_distance_cost >= 0:  # mse loss of two logits vector
                res_loss = args.logit_distance_cost * residual_logit_criterion(
                    class_logit, cons_logit) / minibatch_size
                meters.update('res_loss', res_loss.item())
            else:
                res_loss = 0

            ema_class_loss = class_criterion(ema_logit, target_var)
            ema_class_loss = ema_class_loss.sum() / minibatch_size
            meters.update('ema_class_loss', ema_class_loss.item())

            if args.consistency:
                consistency_weight = get_current_consistency_weight(epoch, args)  # ramp up
                meters.update('cons_weight', consistency_weight)
                consistency_loss = consistency_weight * consistency_criterion(
                    cons_logit, ema_logit) / minibatch_size
                meters.update('cons_loss', consistency_loss.item())
            else:
                consistency_loss = 0
                meters.update('cons_loss', 0)

        else:
            class_logit, _ = model(input_var)

        loss = class_criterion(class_logit, target_var)
        loss = loss * weight_var.float()
        loss = loss * c_weight_var
        loss = loss.sum() / minibatch_size
        meters.update('class_loss', loss.item())

        if isMT:
            loss = loss + consistency_loss + res_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(
            loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100. - prec5[0], labeled_minibatch_size)

        if isMT:
            ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
            meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_top5', ema_prec5[0], labeled_minibatch_size)
            meters.update('ema_error5', 100. - ema_prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if isMT:
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]'
                  'LR {meters[lr]:.4f}\t'
                  'Time {meters[batch_time]:.3f}\t'
                  'Data {meters[data_time]:.3f}\t'
                  'Class {meters[class_loss]:.4f}\t'
                  'Prec@1 {meters[top1]:.3f}\t'
                  'Prec@5 {meters[top5]:.3f}'.format(epoch, i, len(train_loader), meters=meters))

    return meters, global_step


@torch.no_grad()
def validate(eval_loader, model, global_step, epoch, isMT=False):
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()  # use reduction='sum', cuz divide by minibatch_size later

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_var, target_var) in enumerate(eval_loader):
        input_var, target_var = input_var.cuda(), target_var.cuda()
        meters.update('data_time', time.time() - end)
        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        if isMT:
            output1, _, _ = model(input_var)
        else:
            output1, _ = model(input_var)

        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'.format(top1=meters['top1'],
                                                                   top5=meters['top5']))

    return meters['top1'].avg, meters['top5'].avg


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):

    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)
    # print(f'label ratio: {labeled_minibatch_size}/{len(target)}')

    _, pred = output.topk(maxk, 1, True, True)  # (B,5)
    pred = pred.t()  # (5,B)
    target = target.view(1, -1).expand_as(pred)  # (1,B) -> (5,B)
    correct = pred.eq(target)  #  (5,B)

    res = []
    for k in topk:  # (1, 5)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / float(labeled_minibatch_size)))
    return res


def extract_features(train_loader, model, isMT=False):
    """extract features of whole dataset

    Args:
        train_loader ([type]): noshuffle dataloader of full data
        model ([type]): [description]
        isMT (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    model.eval()
    embeddings_all, labels_all = [], []

    # lp.db_semisuper.DatasetFolder.__getitem__()
    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        if isMT:  # mt uses data.TransformTwice, and get a list with 2 items
            X = batch_input[0]
        else:
            X = batch_input

        if isMT:
            _, _, feats = model(X.cuda())
        else:
            _, feats = model(X.cuda())

        embeddings_all.append(feats.data.cpu())  # (200,128)
        labels_all.append(target.data.cpu())  # (200,)

    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())  # (n,128)
    labels_all = torch.cat(labels_all).numpy()  # (n,)
    return embeddings_all, labels_all


def load_args(args, isMT=False):
    label_dir = 'data-local'

    if args.dataset == "cifar100":
        args.batch_size = 128
        args.lr = 0.2
        args.test_batch_size = args.batch_size

        args.epochs = 180
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (
            label_dir, args.dataset, args.num_labeled, args.label_split)
        args.arch = 'cifar_cnn'

    elif args.dataset == "cifar10":

        args.test_batch_size = args.batch_size
        args.epochs = 180
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.arch = 'cifar_cnn'

        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (
            label_dir, args.dataset, args.num_labeled, args.label_split)

    elif args.dataset == "miniimagenet":

        args.train_subdir = 'train'
        args.evaluation_epochs = 30

        args.epochs = 180
        args.batch_size = 128
        args.lr = 0.2
        args.test_batch_size = args.batch_size

        args.epochs = 180
        args.lr_rampdown_epochs = 210
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (
            label_dir, args.dataset, args.num_labeled, args.label_split)
        args.arch = 'resnet18'

    else:
        sys.exit('Undefined dataset!')

    if isMT:
        args.double_output = True
    else:
        args.double_output = False

    return args


import json


def write_args(args: dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(args, f, indent=2)  # auto line-break