#!/usr/bin/env python3
import numpy as np


def specificity(confusion):
    tp = np.diag(confusion)

    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp

    total = np.sum(confusion)
    tn = total - (tp + fp + fn)

    return np.divide(tn, tn + fp, where=(tn + fp) != 0)
