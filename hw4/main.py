#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from svm import SVM
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, datasets


def linearSVM():
    iris = load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("Sepal Width")
    plt.ylabel("Petal Length")
    plt.title("Iris")
    x_min = int(min(X[:, 0])) - 1
    x_max = int(max(X[:, 0])) + 1
    s = SVM()
    s.fit(X, y)
    w, w_0, sp_idx = s.coef_

    X_sp = X[sp_idx]
    y_sp = iris.target[sp_idx]
    plt.scatter(X_sp[:, 0], X_sp[:, 1], c="red")

    x = np.array([i for i in range(x_min, x_max)])
    y = (-w_0 - x*w[0,0]) / w[1, 0]
    y_1 = (1-w_0 - x*w[0,0]) / w[1, 0]
    y_m_1 = (-1-w_0 - x*w[0,0]) / w[1, 0]
    plt.plot(x, y, label="Wx + W_0 = 0")
    plt.plot(x, y_1, label="Wx + W_0 = 1")
    plt.plot(x, y_m_1, label="Wx + W_0 = -1")
    plt.legend(loc=0, borderaxespad=0.)
    plt.show()


def nonlinearSVM():
    data = pd.read_csv("dataset/fourclass.csv")
    X = data[["x1", "x2"]].as_matrix()
    m, n = X.shape
    y = data["y"].as_matrix()
    x_min = int(min(X[:, 0])) - 1
    x_max = int(max(X[:, 0])) + 1

    s = SVM(kernel="polynomial")
    s.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()



if __name__ == "__main__":
    # linearSVM()
    nonlinearSVM()
