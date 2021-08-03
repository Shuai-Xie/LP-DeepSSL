import numpy as np

# Prec@1, Prec@5
exps = {
    'full': [51.640, 87.410],
    'ss': [67.540, 94.630],
    'mt': [68.600, 96.250],
    'mt_ema': [68.630, 96.120],
    'mt_ss': [71.640, 96.610],
    'mt_ss_ema': [71.650, 96.640]
}

for k, v in exps.items():
    print(k, 100 - np.array(v))
