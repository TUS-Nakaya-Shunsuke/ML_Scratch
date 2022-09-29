import numpy as np
import random

def train_test_split(x, y, test_size=0.2, shuffle=True, random_state=None):
    assert len(x) == len(y), "Lengths of x, y must be same!"
    np.random.seed(seed=random_state)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    train_idx = idx[: -int(len(x)*test_size)]
    test_idx = idx[-int(len(x)*test_size): ]
    train_x = x[train_idx]
    test_x = x[test_idx]
    train_y = y[train_idx]
    test_y = y[test_idx]
    return train_x, test_x, train_y, test_y