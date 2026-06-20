#!/usr/bin/env python3
import numpy as np


def sensitivity(confusion):
    # True positives = diagonal
    tp = np.diag(confusion)

    # Actual totals = row sums
    row_sums = np.sum(confusion, axis=1)

    # avoid division by zero
    sensitivity = np.divide(tp, row_sums, where=row_sums != 0)

    return sensitivity
