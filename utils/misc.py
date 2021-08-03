import time
import numpy as np
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_list_to_txt(a_list, outpath):
    mkdir(os.path.dirname(outpath))
    with open(outpath, 'w') as f:
        for a in a_list:
            f.write(f'{a}\n')


def exe_time(name='hi'):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t1 = time.time()
            fn(*args, **kwargs)
            t2 = time.time()
            print(f'{name} time: {round(t2 - t1, 3)}')

        return wrapper

    return decorator


def norm(a, ord=2):
    return np.linalg.norm(a, ord=ord)


def normalize(a, order=2, axis=-1):
    l = np.linalg.norm(a, ord=order, axis=axis, keepdims=True)  # keep the dim, for broadcasting
    l[l == 0] = 1
    return a / l


def cossim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def l2_distance(a, b, use_norm=False):
    if use_norm:
        a, b = normalize(a), normalize(b)
    return norm(a - b)
