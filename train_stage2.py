# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

from helpers import *

args = None
best_prec1 = 0
global_step = 0


def main():
    global global_step
    global best_prec1

    # Name of the model to be trained
    if args.isMT:  # add ss
        model_name = '%s_label%d_mt_ss_split%d_isL2_%d' % (args.dataset, args.num_labeled,
                                                           args.label_split, int(args.isL2))
    else:
        model_name = '%s_label%d_ss_split%d_isL2_%d' % (args.dataset, args.num_labeled,
                                                        args.label_split, int(args.isL2))

    checkpoint_path = 'models/%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # save args
    args_file = '%s/args.json' % checkpoint_path
    write_args(vars(args), args_file)  # args order same to parser.add_arguments

    # save logs
    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'a')

    # Create the dataset and loaders
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    # train_loader_noshuff: twice batchsize full data
    # train_data: lp.db_semisuper.DBSS dataset
    train_loader, eval_loader, train_loader_noshuff, train_data = create_data_loaders(
        **dataset_config, args=args)

    # Create the model
    # if args.isL2=True, normalize feature vector before fc
    model = create_model(num_classes, args)

    # If Mean Teacher is turned on, create the ema model
    if args.isMT:
        ema_model = create_model(num_classes, args, ema=True)  # ema_model, detach params

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    cudnn.benchmark = True

    # Name of the model trained in Stage 1
    if args.isMT:  # checkpoint.180.ckpt, last model
        resume_fn = 'models/%s_label%d_mt_split%d_isL2_%d/checkpoint.180.ckpt' % (
            args.dataset, args.num_labeled, args.label_split, int(args.isL2))
    else:
        resume_fn = 'models/%s_label%d_split%d_isL2_%d/checkpoint.180.ckpt' % (
            args.dataset, args.num_labeled, args.label_split, int(args.isL2))

    # Load the model from Stage 1
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']
    # model
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr  # reset optimizer lr

    # Compute the starting accuracy
    prec1, prec5 = validate(eval_loader, model, global_step, args.start_epoch, isMT=args.isMT)
    if args.isMT:
        ema_prec1, ema_prec5 = validate(eval_loader,
                                        ema_model,
                                        global_step,
                                        args.start_epoch,
                                        isMT=args.isMT)

    print('Resuming from:%s' % resume_fn)

    for epoch in range(args.start_epoch, args.epochs):
        # Extract features and update the pseudo labels
        print('Extracting features...')
        # forward whole dataset each epoch!
        t1 = time.time()
        feats, _ = extract_features(train_loader_noshuff, model, isMT=args.isMT)
        print('extract feature time: %.2f seconds' % (time.time() - t1))
        # iterate 20 times, get propagated labels
        t1 = time.time()
        sel_acc = train_data.update_plabels(feats, k=args.dfs_k, max_iter=20)
        print('selection accuracy: %.2f' % (sel_acc))
        print('label propagation time: %.2f seconds' % (time.time() - t1))

        #  Train for one epoch with the new pseudolabels
        if args.isMT:
            train_meter, global_step = train(
                train_loader,
                model,
                optimizer,
                epoch,
                global_step,
                args,
                ema_model=ema_model,
            )
        else:
            train_meter, global_step = train(
                train_loader,
                model,
                optimizer,
                epoch,
                global_step,
                args,
            )

        # Evaluate the model
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            print("Evaluating the primary model:")
            prec1, prec5 = validate(eval_loader, model, global_step, epoch + 1, isMT=args.isMT)

            if args.isMT:
                print("Evaluating the EMA model:")
                ema_prec1, ema_prec5 = validate(eval_loader,
                                                ema_model,
                                                global_step,
                                                epoch + 1,
                                                isMT=args.isMT)
                is_best = ema_prec1 > best_prec1
                best_prec1 = max(ema_prec1, best_prec1)
            else:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
        else:
            is_best = False

        # Write to the log file and save the checkpoint
        if args.isMT:
            log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                      (epoch, train_meter['class_loss'].avg, train_meter['lr'].avg,
                       train_meter['top1'].avg, train_meter['top5'].avg, prec1, prec5, ema_prec1,
                       ema_prec5))
            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, checkpoint_path, epoch + 1)

        else:
            log.write('%d,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n' % (
                epoch,
                train_meter['class_loss'].avg,
                train_meter['lr'].avg,
                train_meter['top1'].avg,
                train_meter['top5'].avg,
                prec1,
                prec5,
            ))
            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, checkpoint_path, epoch + 1)


if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()

    # Set the other settings
    args = load_args(args, isMT=args.isMT)

    # Use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' %
          (args.num_labeled, args.label_split, args.gpu_id))

    main()
