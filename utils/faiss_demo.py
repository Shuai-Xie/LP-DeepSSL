"""
https://github.com/facebookresearch/faiss/wiki/Getting-started

Test:
nb = 10000, nq = 100
IndexFlatL2_gpu time: 1.420
IndexFlatIP_gpu time: 0.207
"""
import sys

sys.path.insert(0, '/export/nfs/xs/codes/lp-deepssl')

import faiss
import numpy as np
from utils.misc import exe_time

d = 64  # dimension
nb = 10000  # database size
cpu_num = 10
gpu_num = 100
np.random.seed(1234)  # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.  # note 因为数据是随机生成的，第1列加入 index/1000，使得不同? 合理性检测？

# l2-norm
faiss.normalize_L2(xb)


@exe_time('IndexFlatL2')
def IndexFlatL2():
    # build the index
    index = faiss.IndexFlatL2(d)  # brute-force L2 distance search
    index.add(xb)
    k = 4
    D, I = index.search(xb[:cpu_num], k)  # for each row in xq, find knn in xb
    print(I)
    print(D)


@exe_time('IndexFlatIP')
def IndexFlatIP():
    # build the index
    index = faiss.IndexFlatIP(d)  # brute-force L2 distance search
    index.add(xb)
    k = 4
    D, I = index.search(xb[:cpu_num], k)  # for each row in xq, find knn in xb
    print(I)
    print(D)


@exe_time('IndexFlatL2_gpu')
def IndexFlatL2_gpu():
    resources = faiss.StandardGpuResources()  # use a single gpu

    index = faiss.IndexFlatL2(d)  # cpu index
    index = faiss.index_cpu_to_gpu(resources, device=0, index=index)  # gpu index
    index.add(xb)

    k = 4
    D, I = index.search(xb[:gpu_num], k)
    print(I[:5])
    print(D[:5])


@exe_time('IndexFlatIP_gpu')
def IndexFlatIP_gpu():  # 100
    resources = faiss.StandardGpuResources()  # use a single gpu

    index = faiss.IndexFlatIP(d)  # cpu index
    index = faiss.index_cpu_to_gpu(resources, device=0, index=index)
    index.add(xb)

    k = 4
    D, I = index.search(xb[:gpu_num], k)
    print(I[:5])
    print(D[:5])


@exe_time('ssl_IndexFlatIP_gpu')
def ssl_IndexFlatIP_gpu():
    # build GPU index
    resources = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(resources, d, flat_config)  # pass dim=d
    index.add(xb)

    # search
    k = 4
    D, I = index.search(xb[:gpu_num], k)  # return np.ndarray
    print(I[:5])
    print(D[:5])


if __name__ == '__main__':
    # IndexFlatL2_gpu()
    # IndexFlatIP_gpu()
    ssl_IndexFlatIP_gpu()