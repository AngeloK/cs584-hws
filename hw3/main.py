#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from lgregression import LogisticRegression, KClassLogisticRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from evaluator import Evaluator
from sklearn.preprocessing import PolynomialFeatures


if __name__ == "__main__":
    iris = load_iris()
    # print iris
    # X = iris.data[:100, :2]
    # X = iris.data[:100, :]
    # x_min = int(min(X[:, 0])) - 1
    # x_max = int(max(X[:, 0])) + 1
    # y = iris.target[:100]
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.xlabel("sepal length")
    # plt.ylabel("sepal width")
    # plt.show()
    # kf = KFold(X.shape[0], n_folds=10, shuffle=True)

    # polynomial_matrix = PolynomialFeatures(degree = 1)
    # X = polynomial_matrix.fit_transform(X)
    # maximal_acc = 0
    # best_lg = None
    # for train_index, test_index in kf:
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        # l = LogisticRegression()
        # l.fit(X_train, y_train, 0.001, 500)
        # print "Y_predict"
        # predicted = pd.DataFrame(l.predict(X_test), columns=["class"])
        # print "Y_test"
        # test = pd.DataFrame(y_test, columns=["class"])
        # print "\n"
        # e = Evaluator(predicted, test, 1)
        # e.score()
        # if e.accuracy > maximal_acc:
            # maximal_acc = e.accuracy
            # best_lg = l
    # predicted_total = pd.DataFrame(l.predict(X), columns=["class"])
    # y_total = pd.DataFrame(y, columns=["class"])
    # e_total = Evaluator(predicted_total, y_total, 1)
    # print "Result of predicting the whole dataset:"
    # e_total.score()
    # coef = best_lg.coef_
    # coef = coef.T
    # x = np.array([i for i in range(x_min, x_max)])
    # y_ = (-coef[0, 0] - coef[0, 1] * x) / coef[0, 2]
    # plt.plot(x, y_)
    # plt.show()

    X = iris.data
    y = iris.target
    x_min = int(min(X[:, 0]))
    x_max = int(max(X[:, 0]))+1
    kf = KFold(X.shape[0], n_folds=10, shuffle=True)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")

    polynomial_matrix = PolynomialFeatures(degree = 1)
    X = polynomial_matrix.fit_transform(X)
    maximal_acc = 0
    best_kl = None
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        kl = KClassLogisticRegression()
        kl.fit(X, y, 0.001, 10000)
        test = pd.DataFrame(y_test, columns=["class"])
        predicted = pd.DataFrame(kl.predict(X_test), columns=["class"])

        print "=============================="
        print "Test_label"
        print test
        print "------------------------------"
        print "Predicted_label"
        print predicted
        print "=============================="
        # l = LogisticRegression()
        # l.fit(X_train, y_train, 0.001, 500)
        # print "Y_predict"
        # predicted = pd.DataFrame(l.predict(X_test), columns=["class"])
        # print "Y_test"
        # test = pd.DataFrame(y_test, columns=["class"])
        # print "\n"
        e = Evaluator(predicted, test, 1, 3)
        print e.score()
        if e.accuracy > maximal_acc:
            maximal_acc = e.accuracy
            best_kl = kl

    predicted_total = pd.DataFrame(kl.predict(X), columns=["class"])
    y_total = pd.DataFrame(y, columns=["class"])
    e_total = Evaluator(predicted_total, y_total, 1, 3)
    print "Result of predicting the whole dataset:"
    print e_total.score()

    #######################################
    # Plot Classification Boundrary for k-class
    #######################################
    # coefs = best_kl.coef_
    # x = np.array([i for i in range(x_min, x_max)])
    # for coef in coefs:
        # coef = coef.T
        # y_ = (-coef[0, 0] - coef[0, 1] * x) / coef[0, 2]
        # plt.plot(x, y_)
    # plt.show()
    #######################################

    '''
    Sample Output:
    ==============================
    Test_label
        class
    0       0
    1       0
    2       0
    3       1
    4       1
    5       1
    6       1
    7       1
    8       1
    9       2
    10      2
    11      2
    12      2
    13      2
    14      2
    ------------------------------
    Predicted_label
        class
    0       0
    1       0
    2       0
    3       2
    4       1
    5       1
    6       1
    7       1
    8       1
    9       2
    10      1
    11      1
    12      1
    13      2
    14      1
    ==============================
    Accuracy = 0.67
    0  1  2
    0  3  0  0
    1  0  5  1
    2  0  4  2
    ...
    (10-folder cross validation)
    ...

    Result of predicting the whole dataset:
    Accuracy = 0.81
        0   1   2
    0  49   1   0
    1   1  33  16
    2   0  10  40
    '''
