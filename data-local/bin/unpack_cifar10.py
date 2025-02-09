import re
import os
import pickle
import sys

from tqdm import tqdm
from torchvision.datasets import CIFAR10
import matplotlib.image
import numpy as np


# sys.argv[1]: data-local/workdir
# sys.argv[2]: data-local/images/cifar/cifar10/by-image/
work_dir = os.path.abspath(sys.argv[1])
test_dir = os.path.abspath(os.path.join(sys.argv[2], 'test'))
train_dir = os.path.abspath(os.path.join(sys.argv[2], 'train+val'))

cifar10 = CIFAR10(work_dir, download=True)


def load_file(file_name):
    with open(os.path.join(work_dir, cifar10.base_folder, file_name), 'rb') as meta_f:
        return pickle.load(meta_f, encoding="latin1")


def unpack_data_file(source_file_name, target_dir, start_idx):
    print("Unpacking {} to {}".format(source_file_name, target_dir))
    data = load_file(source_file_name)
    for idx, (image_data, label_idx) in tqdm(enumerate(zip(data['data'], data['labels'])), total=len(data['data'])):
        subdir = os.path.join(target_dir, label_names[label_idx])
        name = "{}_{}.png".format(start_idx + idx, label_names[label_idx])
        os.makedirs(subdir, exist_ok=True)
        image = np.moveaxis(image_data.reshape(3, 32, 32), 0, 2)
        matplotlib.image.imsave(os.path.join(subdir, name), image)
    return len(data['data'])


# load class names from batches.meta
label_names = load_file('batches.meta')['label_names']
print("Found {} label names: {}".format(len(label_names), ", ".join(label_names)))
# Found 10 label names: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

start_idx = 0
for source_file_path, _ in cifar10.test_list:
    start_idx += unpack_data_file(source_file_path, test_dir, start_idx)

start_idx = 0
for source_file_path, _ in cifar10.train_list:
    start_idx += unpack_data_file(source_file_path, train_dir, start_idx)
