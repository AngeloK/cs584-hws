#!/usr/bin/env python
# -*- coding: utf-8 -*-


from gda import SingleVariateGDA
import numpy as np
import pandas as pd
from tools import cross_validation
from sklearn.cross_validation import KFold


if __name__ == "__main__":

    base_path = "/Users/Neyanbhbin/Documents/code/Data Analysis/cs584/hw2/dataset/"
    sv = SingleVariateGDA()
    data = pd.read_csv(base_path+"iris/2-d-iris.csv")
    kf = KFold(100, n_folds=10, shuffle=True)
    for train_index, test_index in kf:
        print "===="
        train_data, test_data = data.ix[train_index], data.ix[test_index]
        sv.train(train_data, -1)
        sv.confusion_matrix(sv.classify(test_data), test_data.as_matrix()[:, -1])
        sv.perform()
        print "===="
        print "\n"
