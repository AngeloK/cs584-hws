#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from svm import SVM
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, datasets

'''
Data source url:
    http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex8/ex8.html
'''

def linearSVM():
    iris = load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("Sepal Width")
    # plt.ylabel("Petal Length")
    plt.ylabel("Sepal Length")
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


def radial_SVM():
    iris = load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel("Sepal Width")
    plt.ylabel("Petal Length")
    plt.title("Iris")
    s = SVM(kernel="radial", gamma=0.9)
    s.fit(X, y)
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    Z = s.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.9)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def nonlinearSVM():
    data = pd.read_csv("dataset/fourclass.csv")
    X = data[["x1", "x2"]].as_matrix()
    m, n = X.shape
    y = data["y"].as_matrix()
    h = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    fig_idx = 1
    for d in [10, 20, 30,40, 50, 60]:
        s = SVM(kernel="polynomial", degree=1)
        # s = svm.SVC(kernel='poly', degree=d, C=1.0).fit(X, y)
        # s = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(X, y)
        s.fit(X, y)
        Z = s.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # plt.subplot(3,2, fig_idx)
        # plt.title("Degree=1" %d)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        fig_idx += 1
        break
    plt.show()


if __name__ == "__main__":
    linearSVM()
    # radial_SVM()
    # nonlinearSVM()
