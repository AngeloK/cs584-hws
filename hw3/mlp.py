#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from lgregression import sigmoid, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


class MLP(object):

    def __init__(self):
        self.attr_count = 0
        self.leaning_rate = 0

    def _initialize_parameter(self, X, y):
        '''
        :m the number of samples
        :n the number of features.
        :R the number of neurons
        X is a n * m matrix
        '''
        self.m_sample, self.attr_count = X.shape

        # Define the number of neurons in hidden layer.
        self.neurons_count = int((self.output + self.attr_count) / 2)
        print "Count of neurous = %d" %self.neurons_count

        self.parameter_w = np.random.random((self.neurons_count, self.attr_count))
        self.parameter_v = np.random.random((self.output, self.neurons_count + 1))
        print "shape of v is %s" %str(self.parameter_v.shape)

        # Store the last parameters for momentum
        self.parameter_w_last = np.copy(self.parameter_w)
        self.parameter_v_last = np.copy(self.parameter_v)

        self.y_indicator_matrix = np.empty((self.m_sample, self.output))
        for idx, c in enumerate(self._class_list):
            y_ = self.indicator(y, c)
            self.y_indicator_matrix[:, idx] = y_

    def _split_class(self, y):
        from collections import Counter
        # Using build-in module to compute the count of each class.
        class_count = Counter(y)
        self._class_list = [k for k in class_count.keys()]
        self.output = len(self._class_list)

    def indicator(self, y, label):
        new_y = []
        for l in y:
            if l == label:
                new_y.append(1)
            else:
                new_y.append(0)
        return np.array(new_y).T

    def hidden_layer(self, X, w):
        # The dimension of matrix Z is (R + 1) * m. The extra dimension is constant
        # extra 1 dimension for bias.
        Z = sigmoid(np.dot(X, w.T))
        p = PolynomialFeatures(degree = 1)
        Z = p.fit_transform(Z)
        return Z

    def fit(self, X, y, learning_rate, iteration=1):

        # Initialize all parameters.
        self._split_class(y)
        self._initialize_parameter(X, y)
        self.learning_rate = learning_rate


        for i in range(iteration):
            self.parameter_w_last = np.copy(self.parameter_w)
            self.parameter_v_last = np.copy(self.parameter_v)

            print "parameter w last : %s" %str(self.parameter_w_last)
            print "parameter w : %s" %str(self.parameter_w)
            print "parameter v last : %s" %str(self.parameter_v_last)
            print "parameter v : %s" %str(self.parameter_v)

            Z = self.hidden_layer(X, self.parameter_w_last)
            y_hat = sigmoid(np.dot(Z, self.parameter_v_last.T))
            d = y_hat - self.y_indicator_matrix

            # update v
            temp_v = np.empty(self.parameter_v_last.shape)
            for idx in range(self.output):
                d_j = d[:, idx].reshape((self.m_sample, 1))
                v_j = np.sum(d_j * Z, axis=0).reshape((1, self.neurons_count + 1))
                temp_v[idx, :] = v_j
            self.parameter_v -= self.learning_rate * temp_v

            # update w
            temp_w = np.empty(self.parameter_w_last.shape)
            for idx in range(self.neurons_count):
                v_l_j = self.parameter_v_last[:, idx].reshape((self.output, 1))
                Z_j = Z[:, idx].reshape((self.m_sample, 1))
                w_j = np.sum(np.dot(d, v_l_j) * Z_j * (1 - Z_j) * X, axis=0)
                temp_w[idx, :] = w_j
            self.parameter_w -= self.learning_rate * temp_w

    def predict(self, X_test):

        m_test, n = X_test.shape
        polynomial_matrix = PolynomialFeatures(degree = 1)
        predicted = []
        for idx in range(m_test):
            X = X_test[idx, :].reshape((n, 1))
            Z = np.dot(self.parameter_w, X).T
            Z = polynomial_matrix.fit_transform(Z).T
            y_list = sigmoid(np.dot(self.parameter_v, Z))
            print y_list
            y = np.argmax(y_list)
            predicted.append(y)
        return np.array(predicted)
