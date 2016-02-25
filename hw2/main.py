#!/usr/bin/env python
# -*- coding: utf-8 -*-


from gda import SingleVariateGDA
import numpy as np
import pandas as pd
from tools import cross_validation


if __name__ == "__main__":

    base_path = "/Users/Neyanbhbin/Documents/code/Data Analysis/cs584-hw2/dataset/"
    sv = SingleVariateGDA()
    data = pd.read_csv(base_path+"bank/2-d-bank-full.csv").as_matrix()
    te_d, te_s, tr_d, tr_s = cross_validation(data, 1)
    sv.train(tr_d)
    print sv.mean_
    print sv.variance_
    print sv.precision(sv.classify(te_d[:, :-1]), te_d[:, -1])
