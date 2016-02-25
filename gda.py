#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


class GaussianDistributionAnalysis(object):

    def __init__(self):
        self.mean_ = None
        self.variance_ = None

    def compute_parameters(self, training_data):
        # Import data
        self._x = training_data[:, :-1]
        self._y = training_data[:, -1]
        self.n_sample = self._x.shape[0]

        # Compute parameters
        self._compute_main()
        self._compute_variance()
        self._split_class()

    def _compute_main(self):
        # If axis=1, then this function will compute the mean of column.
        self.mean_ = np.mean(self._x, axis=1)

    def _split_class(self):
        from collections import Counter
        # Compute of the number each class apppearing in training data set using
        # buildin python tool
        class_count = Counter(self._y)
        self._class_list = [k for k in class_count.keys()]

    def _compute_variance(self):
        # If the training data is 1-d dataset, then this will compute the
        # variance if feature x, else this function will compute the covariance
        # matrix
        self.variance_ = np.cov(self._x)

    def prior(self, class_):
        # Compute the possibility of p(y=classj)
        count = 0
        for sample_index in range(self.n_sample):
            try:
                if self._y[sample_index] == class_:
                    count += 1
            except:
                # Ingore invalid value
                pass
        return count/self.n_sample

    def predict(self, testing_data):
        # Initialize a new empty np array to store predicted value
        n_testing_sample = testing_data.shape[0]
        predicted_class = np.empty(n_testing_sample, 1)
        for sample_index in range(n_testing_sample):
            predicted_class[sample_index] = self.classify(testing_data[sample_index, :])
        return predicted_class


class SingleVariateGDA(GaussianDistributionAnalysis):

    def __init__(self):
        pass

    def train(self, training_data):
        self.compute_parameters(training_data)

    def likelihood(self, x, class_):
        p_x_given_y = np.log(
            (1/(np.sqrt(2 * np.pi))) * np.exp((-1/(2*self.variance_))*(x - self.mean_) ** 2)
        ) + np.log(self.prior(class_))
        return p_x_given_y

    def classify(self, testing_sample):
        maximum_likelihood = np.ninf
        predicted_class = None
        for c in self._class_list:
            p = self.likelihood(testing_sample, c)
            if p > maximum_likelihood:
                maximum_likelihood = p
                predicted_class = c
        return predicted_class

    def perform(self, testing_class):
        pass


class MultivariateGDA(GaussianDistributionAnalysis):

    def __init__(self):
        pass
