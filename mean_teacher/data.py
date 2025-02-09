# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.
"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler

import pdb
import lp.db_semisuper

LOG = logging.getLogger('main')
NO_LABEL = -1


class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """
    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation, ypad - ytranslation,
                                    xpad + xsize - xtranslation, ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TransformOnce:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        # out2 = self.transform(inp)
        return out1


def relabel_dataset(dataset: lp.db_semisuper.DBSS, labels: dict):
    """relabel label/unlabel images and set below attrs of dataset.
        all_labels, p_labels, labeled_idx, unlabeled_idx

    Args:
        dataset (lp.db_semisuper.DBSS): a subclass of DatasetFolder
        labels (dict): {img_name: cls_name}

    Raises:
        LookupError: [description]

    Returns:
        labeled_idxs, unlabeled_idxs: two list of label/unlabel idxs
    """
    unlabeled_idxs = []

    # split label/unlabel images based on `labels` dict
    for idx in range(len(dataset.imgs)):
        path, orig_label = dataset.imgs[idx]  # path, cls2idx
        filename = os.path.basename(path)
        dataset.all_labels.append(orig_label)
        dataset.p_labels.append(-1)  # -1 init
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            dataset.labeled_idx.append(idx)
            del labels[filename]  # save memory and speed query
        else:
            dataset.imgs[idx] = path, NO_LABEL  # path, -1
            dataset.unlabeled_idx.append(idx)
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))
    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
        and merge imgs from two sets into one batch

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through as many times as needed. 
        cuz num_secondary << num_primary
    """
    def __init__(self, primary_indices: list, secondary_indices: list, batch_size: int,
                 secondary_batch_size: int):
        """
        Args:
            primary_indices (list): unlabel images are primary
            secondary_indices (list): label images are secondary
            batch_size (int): batch size
            secondary_batch_size (int): label img number in a batch
        """
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (primary_batch + secondary_batch for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
        )) # Tuple[List]

    def __len__(self):  # An 'epoch' is one iteration through the primary indices.
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
