#!/usr/bin/env python3
import numpy as np


def precision(confusion):
    tp = np.diag(confusion)
    col_sums = np.sum(confusion, axis=0)

    return np.divide(tp, col_sums, where=col_sums != 0)
