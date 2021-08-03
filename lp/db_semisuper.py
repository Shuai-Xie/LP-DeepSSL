# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

import os
import os.path
import pdb
import pickle
import sys
import time

import faiss
import numpy as np
import scipy
import scipy.stats
import torch
import torch.utils.data as data
from PIL import Image

from .diffusion import *
from utils.misc import normalize


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


# ref: torchvision.datasets.ImageFolder
class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]  # s = (sample path, class_index)

        self.transform = transform
        self.target_transform = target_transform

        imfile_name = '%s/images.pkl' % self.root  # train_subdir/images.pkl
        if os.path.isfile(imfile_name):
            with open(imfile_name, 'rb') as f:
                self.images = pickle.load(f)
        else:
            self.images = None

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            attrs from subclass: p_labels, p_weights, class_weights

        Returns:
            tuple: (sample, target, weight, c_weight)
                  target: class_index of the target class
                  weight: pseudo sample weight of this sample
                c_weight: class weight of this sample
        """

        path, target = self.samples[index]

        if (index not in self.labeled_idx):
            target = self.p_labels[index]  # pseudo label

        weight = self.p_weights[index]  # pseudo weight

        if self.images is not None:
            sample = Image.fromarray(self.images[index, :, :, :])  # read from pkl
        else:
            sample = self.loader(path)  # read from img

        if self.transform is not None:
            sample = self.transform(sample)

        c_weight = self.class_weights[target]

        return sample, target, weight, c_weight

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:  # default is PIL
        return pil_loader(path)


class DBSS(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(DBSS, self).__init__(root,
                                   loader,
                                   IMG_EXTENSIONS,
                                   transform=transform,
                                   target_transform=target_transform)
        self.imgs = self.samples

        self.pos_list = dict()
        self.pos_w = dict()
        self.pos_dist = dict()

        self.labeled_idx = []
        self.unlabeled_idx = []
        self.all_labels = []  # save ori label idx

        # pseudo weights and cls weights
        self.p_labels = []
        self.p_weights = np.ones((len(self.imgs), ))  # default 1
        self.class_weights = np.ones((len(self.classes), ),
                                     dtype=np.float32)  # default cls_weight = 1

        self.images_lists = [[] for i in range(len(self.classes))]  # each cls has a list

    def update_plabels(self, X, k=50, max_iter=20):
        """update pseudo lables

        Args:
            X (np.ndarray): feature vectors (n,128)
            k (int, optional): neighborhood size. Defaults to 50. [hyperparam]
            max_iter (int, optional): iterate times to get Z. Defaults to 20. [hyperparam]

        Returns:
            [type]: [description]
        """

        print('Updating pseudo-labels...')
        alpha = 0.99

        # label/unlabel index
        labels = np.asarray(self.all_labels)  # (N,)
        labeled_idx = np.asarray(self.labeled_idx)  # (L,)
        unlabeled_idx = np.asarray(self.unlabeled_idx)  # (N-L,)

        # kNN search for the graph with faiss
        N, d = X.shape  # note this N is the set of samples to be propapated, not len(labels) necessarily
        # build index
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index
        faiss.normalize_L2(X)  # note L2 norm, then L2 = IP = cos similarity
        index.add(X)
        # search index
        c = time.time()
        D, I = index.search(X, k + 1)  # use k+1, cuz the 1st nearest is itself
        elapsed = time.time() - c
        print('kNN Search done in %d seconds' % elapsed)

        # Create the graph
        D = D[:, 1:]**3  # note (eq 9)
        I = I[:, 1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx, (k, 1)).T  # (N, k), row index repeat
        # sparse weight matrix, k/N has affinity weights
        W = scipy.sparse.csr_matrix(
            (
                D.flatten('F'),  # data, 'F' column-major flatten
                (row_idx_rep.flatten('F'), I.flatten('F'))  # (row_idx, col_idx)
            ),
            shape=(N, N))
        W = W + W.T  # symmetric afffinity matrix

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())  # W_ii = 0
        S = W.sum(axis=1)
        S[S == 0] = 1  # if sum(sim)=0, attentioned w_ij = w_ij, cuz the whole impact equals zero.
        D = np.array(1. / np.sqrt(S))  # D^(-1/2)
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D  # normalized weight

        # Initiliaze the y vector for each class (eq 5, normalized with the class size) and apply label propagation
        C = len(self.classes)
        Z = np.zeros((N, C))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn  # (I-αW)
        for i in range(C):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]  # sample idx with cls=i
            y = np.zeros((N, ))
            y[cur_idx] = 1.0 / cur_idx.shape[0]  # cls i samples, cls_weight
            # note solve (I-αW)Z = Y (eq 10)
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:, i] = f  # get the propagated matrix of each class

        # Handle numberical errors
        Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11)
        probs_l1 = normalize(Z, order=1, axis=1)  # use l1-norm so that sum(probs)=1
        probs_l1[probs_l1 < 0] = 0
        entropy = scipy.stats.entropy(probs_l1, axis=1)  # (N,c) -> (N,)
        weights = 1 - entropy / np.log(C)  # (eq 11) where log(c) is the max entropy
        weights = weights / np.max(weights)  # max_val normalize
        p_labels = np.argmax(probs_l1, 1)

        # Compute the accuracy of pseudo labels for statistical purposes
        # note this line can be placed after line 350, place here is more strict
        acc = (p_labels == labels).mean()

        p_labels[labeled_idx] = labels[labeled_idx]  # GT labeled still use GT
        weights[labeled_idx] = 1.0

        self.p_weights = weights.tolist()  # pseudo sample weights
        self.p_labels = p_labels

        # Compute the weight for each class, c_i = 1/C * N / N_c
        for i in range(C):
            self.class_weights[i] = (labels.shape[0] / C) / float((self.p_labels == i).sum())

        return acc
