#!/usr/bin/env python3
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision

def f1_score(confusion):
    recall = sensitivity(confusion)
    prec = precision(confusion)

    return np.divide(
        2 * prec * recall,
        prec + recall,
        where=(prec + recall) != 0
    )
