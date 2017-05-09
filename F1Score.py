import numpy as np


def F1Scroe(preY, reallyY):
    tp = np.where((preY == reallyY) and (preY == 0), 1, 0)
    tn = np.where((preY == reallyY) and (preY == 1), 1, 0)
    fp = np.where((preY != reallyY) and (preY == 0), 1, 0)
    fn = np.where((preY != reallyY) and (preY == 1), 1, 0)

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * (p * r) / (p + r)
