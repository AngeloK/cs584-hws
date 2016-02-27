#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv


def normal_equation(points, degrees=1):
    '''
    X*beta = Y
    -> inv(X.T * X) * X.T * Y = beta
    The default value of degree is 1
    '''
    # one_vector is a vector with size (len(data), 1) and all values are 1.
    one_vector = np.ones((len(points), 1), dtype=np.float)
    X_origin = np.matrix(points[:, :-1])
    X = np.matrix(points[:, :-1])
    for degree in range(2, degrees+1):
        X = np.column_stack((X, np.power(X_origin, degree)))
    X = np.column_stack((one_vector, X))
    Y = np.matrix(points[:, -1])
    Y = Y.T
    inv_Xt_X = inv((X.T)*X)
    beta = inv_Xt_X * X.T * Y
    print "beta= %s" % beta
    return beta


def create_polynomial_regression_function(x, parameters):
    # Initialize function
    y = 0
    if parameters.shape[0] < 1:
        return "Parameter Invalid"
    degree = 0
    for coefficient_index in range(parameters.shape[0]):
        print parameters.item((coefficient_index, 0))
        y += parameters.item((coefficient_index, 0)) * np.power(x, degree)
        degree += 1
    return y
