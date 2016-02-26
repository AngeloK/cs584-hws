#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
np.set_printoptions(threshold=np.inf)


class DataLengthNotMatchError(Exception):

    def __init__(self):
        super(DataLengthNotMatchError, self).__init__(
            "The length of predicted data and testing data is not matching"
        )


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

    def _split_class(self):
        from collections import Counter
        # Compute of the number each class apppearing in training data set using
        # buildin python tool
        class_count = Counter(self._y)
        self._class_list = [k for k in class_count.keys()]

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
        # print "class %s %s" % (class_, str(count))
        return count/self.n_sample

    def predict(self, testing_data):
        # Initialize a new empty np array to store predicted value
        n_testing_sample = testing_data.shape[0]
        predicted_class = np.empty(n_testing_sample, 1)
        for sample_index in range(n_testing_sample):
            predicted_class[sample_index] = self.classify(
                testing_data[
                    sample_index,
                    :])
        return predicted_class


    def is_parameter_valid(self, predicted_class, testing_class):
        if predicted_class.shape[0] != testing_class.shape[0]:
            return False
        return True

    def precision(self, predicted_class, testing_class):
        if not self.is_parameter_valid(predicted_class, testing_class):
            raise DataLengthNotMatchError

        n_testing_sample = predicted_class.shape[0]
        accurate_count = 0
        for i in range(n_testing_sample):
            if predicted_class[i] == testing_class[i]:
                accurate_count += 1
        return accurate_count/n_testing_sample

    def recall(self, predicted_class, testing_class):
        pass

class SingleVariateGDA(GaussianDistributionAnalysis):

    def __init__(self):
        super(SingleVariateGDA, self).__init__()

    def train(self, training_data):
        self.compute_parameters(training_data)

    def _compute_main(self):
        # If axis=0, then this function will compute the mean of column.
        self.mean_ = np.mean(self._x, axis=0)[0]

    def _compute_variance(self):
        # If the training data is 1-d dataset, then this will compute the
        # variance if feature x, else this function will compute the covariance
        # matrix
        self.variance_ = np.var(self._x, axis=0)[0]

    def likelihood(self, x, class_):
        p_x_given_y = np.log(
            (1/(np.sqrt(2 * np.pi) * self.variance_)) * np.exp((-1/(2*self.variance_))*(x - self.mean_) ** 2)
        ) + np.log(self.prior(class_))
        return p_x_given_y

    def classify(self, testing_sample):
        print "class 'yes' prior: %s" % str(self.prior(self._class_list[0]))
        print "class 'no' prior: %s" % str(self.prior(self._class_list[1]))
        # transform datatype into 'float' for computation
        testing_sample = np.array(testing_sample, dtype=np.float)
        n_sample = testing_sample.shape[0]

        predicted_class = []
        for sample_index in range(n_sample):
            maximum_likelihood = -np.inf
            predicted = ""
            for c in self._class_list:
                p = self.likelihood(testing_sample[sample_index, :], c)
                if p > maximum_likelihood:
                    maximum_likelihood = p
                    predicted = c
            predicted_class.append(predicted)
        # print predicted_class
        return np.array(predicted_class)

    def confusion_matrix(self, predicted_class, testing_class):
        if not self.is_parameter_valid(predicted_class, testing_class):
            raise DataLengthNotMatchError
        # Compute each variable as 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        n_sample = predicted_class.shape[0]
        for sample_index in range(n_sample):
            if testing_class[sample_index] == self._class_list[0]:
                if predicted_class[sample_index] == testing_class[sample_index]:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if predicted_class[sample_index] == testing_class[sample_index]:
                    false_positive += 1
                else:
                    false_negative += 1
        print "TP %d" % true_positive
        print "TN %d" % true_negative
        print "FP %d" % false_positive
        print "FN %d" % false_negative

    def perform(self, testing_class):
        pass


class MultivariateGDA(GaussianDistributionAnalysis):

    def __init__(self):
        pass
