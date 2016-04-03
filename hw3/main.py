#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from lgregression import LogisticRegression, KClassLogisticRegression, sigmoid
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, accuracy_score
from evaluator import Evaluator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_mldata
from matplotlib import cm
from mlp import MLP
from sklearn.neural_network import MLPClassifier
from nolearn.dbn import DBN
import time

mnist = fetch_mldata('MNIST original')
mnist.data.shape
mnist.target.shape
np.unique(mnist.target)

X, y = mnist.data / 255., mnist.target
# X_train, X_test = X[:60000], X[60000:]
# y_train, y_test = y[:60000], y[60000:]
size = len(y)
# size_train=len(y_train)
# size_test=len(y_test)

ind = [ k for k in range(size) if y[k]==0 or y[k]==1]
X = X[ind, :]
y = y[ind]


def two_class_lg(iteration=100):
    global X, y

    # iris = load_iris()
    # print iris
    # X = iris.data[:100, :2]
    # X = iris.data
    x_min = int(min(X[:, 0])) - 1
    x_max = int(max(X[:, 0])) + 1
    # y = iris.target[:100]
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.xlabel("sepal length")
    # plt.ylabel("sepal width")
    # plt.title("Iris")
    # plt.show()
    kf = KFold(X.shape[0], n_folds=10, shuffle=True)

    polynomial_matrix = PolynomialFeatures(degree = 1)
    X = polynomial_matrix.fit_transform(X)
    maximal_acc = 0
    best_lg = None
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        l = LogisticRegression()
        l.fit(X_train, y_train, 0.001, iteration)
        print "Y_predict"
        predicted = pd.DataFrame(l.predict(X_test), columns=["class"])
        print "Y_test"
        test = pd.DataFrame(y_test, columns=["class"])
        print "\n"
        e = Evaluator(predicted, test, 1)
        e.score()
        if e.accuracy > maximal_acc:
            maximal_acc = e.accuracy
            best_lg = l
    predicted_total = pd.DataFrame(l.predict(X), columns=["class"])
    y_total = pd.DataFrame(y, columns=["class"])
    e_total = Evaluator(predicted_total, y_total, 1)
    print "Result of predicting the whole dataset:"
    e_total.score()
    coef = best_lg.coef_
    coef = coef.T
    x = np.array([i for i in range(x_min, x_max)])
    y_ = (-coef[0, 0] - coef[0, 1] * x) / coef[0, 2]
    plt.plot(x, y_, label="iteration=%d" %iteration)
    # plt.show()


if __name__ == "__main__":
    # two_class_lg(100)
    # two_class_lg(500)
    # two_class_lg(1000)
    # two_class_lg(10000)
    # plt.legend(loc=4, borderaxespad=0.)
    # plt.show()

    ############################################################################
    iris = load_iris()
    # # print iris
    # X = iris.data[:100, :2]
    # # X = iris.data
    # x_min = int(min(X[:, 0])) - 1
    # x_max = int(max(X[:, 0])) + 1
    # y = iris.target[:100]
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.xlabel("sepal length")
    # plt.ylabel("sepal width")
    # plt.title("Iris")
    # # plt.show()
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

    # X = iris.data
    # y = iris.target
    # x_min = int(min(X[:, 0]))
    # x_max = int(max(X[:, 0]))+1
    # kf = KFold(X.shape[0], n_folds=10, shuffle=True)
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.xlabel("sepal length")
    # plt.ylabel("sepal width")

    # polynomial_matrix = PolynomialFeatures(degree = 1)
    # X = polynomial_matrix.fit_transform(X)
    # maximal_acc = 0
    # best_kl = None
    # for train_index, test_index in kf:
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        # kl = KClassLogisticRegression()
        # kl.fit(X, y, 0.001, 10000)
        # test = pd.DataFrame(y_test, columns=["class"])
        # predicted = pd.DataFrame(kl.predict(X_test), columns=["class"])

        # print "=============================="
        # print "Test_label"
        # print test
        # print "------------------------------"
        # print "Predicted_label"
        # print predicted
        # print "=============================="
        # l = LogisticRegression()
        # l.fit(X_train, y_train, 0.001, 500)
        # print "Y_predict"
        # predicted = pd.DataFrame(l.predict(X_test), columns=["class"])
        # print "Y_test"
        # test = pd.DataFrame(y_test, columns=["class"])
        # print "\n"
        # e = Evaluator(predicted, test, 1, 3)
        # print e.score()
        # if e.accuracy > maximal_acc:
            # maximal_acc = e.accuracy
            # best_kl = kl

    # predicted_total = pd.DataFrame(kl.predict(X), columns=["class"])
    # y_total = pd.DataFrame(y, columns=["class"])
    # e_total = Evaluator(predicted_total, y_total, 1, 3)
    # print "Result of predicting the whole dataset:"
    # print e_total.score()

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

    # mnist = fetch_mldata('MNIST original')
    # mnist.data.shape
    # mnist.target.shape
    # np.unique(mnist.target)

    X, y = mnist.data / 255., mnist.target
    # X_train, X_test = X[:60000], X[60000:]
    # y_train, y_test = y[:60000], y[60000:]
    # size_train=len(y_train)
    # size_test=len(y_test)
    size = len(y)

    # clf = MLPClassifier(
            # algorithm='sgd',
            # activation='logistic',
            # alpha=1e-2,
            # hidden_layer_sizes=(300, 10),
            # random_state=1
        # )
    # clf.fit(X_train, y_train)
    # p = clf.predict(X_test)
    # classification_report(y_test, p)
    # print clf.score(X_test, y_test)

    idx_0 = [k for k in range(size) if y[k] == 0]
    idx_1 = [k for k in range(size) if y[k] == 1]
    idx_2 = [k for k in range(size) if y[k] == 2]
    ind = idx_0 + idx_1+ idx_2
    training_size = len(ind)

    kf = KFold(training_size, n_folds=10, shuffle=True)
    X = X[ind, :]
    print X
    y = y[ind]

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # start = time.time()
        # m = MLPClassifier(
            # algorithm='sgd',
            # activation='logistic',
            # alpha=1e-5,
            # hidden_layer_sizes=(800, 4),
            # random_state=1
        # )
        # m.fit(X_train, y_train)
        # p = m.predict(X_test)
        # end = time.time()
        # duration = end - start
        # print "Executing time is %s" %str(duration)
        # print m.score(X_test, y_test)
        # print "\n"

        # print X_train[:2, :]
        # print y_train[:2]

        # m = MLP()
        # m.fit(X_train, y_train, 0.0001, 500)

        # p = m.predict(X_test)
        # print classification_report(y_test, p)


        h = [i for i in range(1,1000, 200)]

        dbn = DBN(
            [X_train.shape[1], 200, 3],
            learn_rates = 0.0001 * i,
            learn_rate_decays = 0.9,
            epochs = 10,
            verbose = 1
        )
        dbn.fit(X_train, y_train)
        p = dbn.predict(X_test)
        a = accuracy_score(y_test, p)
        eva = classification_report(y_test, p)
        print "Accuracy = %.2f" %a
        print eva
