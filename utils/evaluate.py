"""
Helper functions.
"""

from numpy.linalg import *
import random
from operator import add

import codecs
import string
import os
import sys
from numpy import *
import numpy as np
from sklearn.metrics import roc_auc_score
floattype = float64
def f_compute(correct, predictions):
    # print 'compute f,p,r \n'

    TP = 0.
    FP = 0.
    FN = 0.
    TN = 0.
    maxac = 0.
    maxf = 0.
    maxr = 0.
    maxp = 0.
    maxTP = 0.
    maxFP = 0.
    maxFN = 0.
    maxTN = 0.

    assert len(predictions) == len(correct)
    pairs = [(predictions[i], correct[i]) for i in range(len(predictions))]
    j = -1.0
    maxj = -1.0
    f_auc=roc_auc_score(correct, predictions)

    while j < 1.0:
        TP = 0.
        FP = 0.
        FN = 0.
        TN = 0.
        for i in range(0, len(pairs)):
            if (pairs[i][0] > j):
                if (pairs[i][1] == 1.0):
                    TP += 1
                else:
                    FP += 1
            else:
                if (pairs[i][1] == 1.0):
                    FN += 1
                else:
                    TN += 1

        if (TP == 0. and (FP == 0. or FN == 0.)):
            F = 0.
            prec = 0
            rec = 0
        else:
            prec = float(TP) / float(TP + FP)
            rec = float(TP) / float(TP + FN)
            if (prec == 0 and rec == 0):
                F = 0.
            else:
                F = (2. * prec * rec) / (prec + rec)
        ac = float(TP + TN) / float(TP + FP + FN + TN)

        if F > maxf:
            maxf = F
            maxr = rec
            maxp = prec
            maxac = ac
            maxj = j
            maxTP = TP
            maxFP = FP
            maxFN = FN
            maxTN = TN
        j += 0.001
    f_f = maxf
    return f_auc, f_f, maxp, maxr, maxac,maxj, maxTP, maxFP, maxTN, maxFN

def f_score(pre, answer):
    TP = 0.
    FP = 0.
    TN = 0.
    FN = 0.
    for i in range(len(answer)):
        if answer[i] == 1.0:
            if pre[i] == 1.0:
                TP += 1
            else:
                FN += 1
        else:
            if pre[i] == 1.0:
                FP += 1
            else:
                TN += 1
    if (TP == 0. and (FP == 0. or FN == 0.)):
        F = 0.
        prec = 0
        rec = 0
    else:
        prec = float(TP) / float(TP + FP)
        rec = float(TP) / float(TP + FN)
        if (prec == 0 and rec == 0):
            F = 0.
        else:
            F = (2. * prec * rec) / (prec + rec)
    return prec, rec, F, TP, FP, TN, FN