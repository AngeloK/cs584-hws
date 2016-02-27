#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from numpy.linalg import inv



def locally_weighted_regression(predicting_x, dataset, k=0.01):
    '''
    predicting_x is a list contains all attribute value
    theta = inv(Z.T * W * Z) * Z.T * W * Y
    W is the weight matrix that only has diagonal elements.
    Here we use Gaussian kernel to compute the weight matrix
    '''
    # Transfer list to np.array for operations
    predicting_x = np.array(predicting_x)
    label = np.matrix(dataset.iloc[:, -1])
    data = np.matrix(dataset.iloc[:, 1:-1])
    col, row = data.shape
    one_vector = np.ones((col, 1), dtype=np.float)
    Z = np.column_stack((one_vector, data))
    W = np.zeros((col, col), dtype=float)
    for data_index in range(0, col):
        train_data = data[data_index]
        w = 0
        for dif in (train_data - predicting_x).tolist()[0]:
            w += dif*dif
        W[data_index][data_index] = np.exp(w/(-2*k*k))
    theta = inv(Z.T * W * Z) * Z.T * W * label.T
    return theta

if __name__ == "__main__":
    data = pd.read_csv("abalone.csv")
    sample_x = data.iloc[299].values[1: -1].astype(float)
    print "Sample_x = %s" % str(sample_x)
    sample_label = data.iloc[299, -1]
    print "Label = %s " % sample_label
    theta = locally_weighted_regression(sample_x, data)
    one_vector = np.ones((1,1), dtype=np.float)
    sample = np.append([1], sample_x)
    print np.matrix(sample) * theta
