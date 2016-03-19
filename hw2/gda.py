#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


class DataLengthNotMatchError(Exception):
    '''
    Exceptions for two dataset unmatch problem.
    '''

    def __init__(self):
        super(DataLengthNotMatchError, self).__init__(
            "The length of predicted data and testing data is not matching"
        )


class GaussianDiscriminantAnalysis(object):

    def __init__(self):
        self.mean_ = None
        self.variance_ = None

    def compute_parameters(self, training_data, class_col_idx):
        # The type of data shoudl be pandas's Dataframe.
        # we use this datatype fot the sake of better filter data based
        # on specific value
        if not isinstance(training_data, pd.core.frame.DataFrame):
            raise TypeError("DataFrame instance needed")
        self.class_col_idx = class_col_idx
        self.class_column_name = training_data.columns[class_col_idx]
        y = training_data.as_matrix()[:, class_col_idx]
        self._y = y
        self.n_sample = y.shape[0]
        self._split_class(y)

    def _split_class(self, y):
        from collections import Counter
        # Compute of the number each class apppearing in training data set using
        # buildin python tool
        class_count = Counter(y)
        self._class_list = [k for k in class_count.keys()]

    def train(self, training_data, class_col_idx):
        # Basic method for training model based on training data.
        self.compute_parameters(training_data, class_col_idx=class_col_idx)

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

    def is_parameter_valid(self, predicted_class, testing_class):
        '''
        Avoid bad input parameters.
        '''
        if predicted_class.shape[0] != testing_class.shape[0]:
            return False
        return True

    def predict(self, testing_sample, threshold=1.0):
        # transform datatype into 'float' for computation
        # testing_sample = np.array(testing_sample, dtype=np.float)
        testing_sample = testing_sample.as_matrix()[:, :-1]
        n_sample = testing_sample.shape[0]

        predicted_class = []
        for sample_index in range(n_sample):
            maximum_likelihood = -np.inf
            predicted = ""
            for c in self._class_list:
                pro = self.likelihood(testing_sample[sample_index, :], c)
                if pro > maximum_likelihood:
                    maximum_likelihood = pro
                    predicted = c
            predicted_class.append(predicted)
        return np.array(predicted_class)

    def confusion_matrix(self, predicted_class, testing_class, selected_class=None):
        # selected_class defined as the value we decided as positive
        if not self.is_parameter_valid(predicted_class, testing_class):
            raise DataLengthNotMatchError
        if not selected_class:
            selected_class = self._class_list[0]

        # Initialize confusion matrix.
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        # Compute each element confusion matrix.
        n_sample = predicted_class.shape[0]
        for sample_index in range(n_sample):
            if testing_class[sample_index] == selected_class:
                if predicted_class[sample_index] == testing_class[sample_index]:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if predicted_class[sample_index] == testing_class[sample_index]:
                    true_negative += 1
                else:
                    false_negative += 1
        print "TP %d" %true_positive
        print "FP %d" %false_positive
        print "TN %d" %true_negative
        print "FN %d" %false_negative


        # In order to avoid zero division problem, we assign the value of
        # precision or recall to NAN if zero division happens.
        if (true_positive + false_positive) == 0:
            self.precision = np.nan
        else:
            self.precision = (true_positive) / (true_positive + false_positive)
        if (true_positive + false_positive) == 0:
            self.recall = np.nan
        else:
            self.recall = true_negative / (true_negative + false_negative)
        if self.precision == np.nan or self.recall == np.nan:
            self.f_measure = np.nan
        else:
            self.f_measure = (2*self.recall * self.precision / \
                                    (self.precision + self.recall))

        print "Accuracy = %.2f" % self.accuracy
        print "Precision = %.2f" % self.precision
        print "Recall = %.2f" % self.recall
        print "F-Measure = %.2f" % self.f_measure


class SingleDimensionTwoClassGDA(GaussianDiscriminantAnalysis):
    '''
    This class is used for training single feature dataset. The different bewteen
    this class and MultiDimensionsGDA is here it only compute variance. But in
    the MultiDimensionsGDA, we need to compute covariance matrix.
    '''

    def __init__(self):
        super(SingleDimensionTwoClassGDA, self).__init__()

    def compute_parameters(self, training_data, class_col_idx):
        super(SingleDimensionTwoClassGDA, self).compute_parameters(training_data, class_col_idx)
        self._compute_main_and_variance(training_data)

    def _compute_main_and_variance(self, training_data):
        self.mean_ = []
        self.variance_ = []
        for idx, c in enumerate(self._class_list):
            class_column_name = training_data.columns[self.class_col_idx]
            self.mean_.append(
                training_data[training_data[class_column_name] == c].mean()[0]
            )
            self.variance_.append(
                training_data[training_data[class_column_name] == c].var()[0]
            )

    def likelihood(self, x, class_):
        # We use log likelihood to compute the similarity of current sample to
        # each class.
        class_idx = self._class_list.index(class_)
        log_likelihood = np.log(1/np.sqrt(2 * np.pi)) -\
            np.log(self.variance_[class_idx]) +\
            ((-1/2) * ((x - self.mean_[class_idx])/self.variance_[class_idx]) ** 2) +\
            np.log(self.prior(class_))
        return log_likelihood


class MultiDimensionsGDA(GaussianDiscriminantAnalysis):

    def __init__(self):
        super(MultiDimensionsGDA, self).__init__()

    def compute_parameters(self, training_data, class_col_idx):
        super(MultiDimensionsGDA, self).compute_parameters(training_data, class_col_idx)
        self._compute_main_and_covariance_matrix(training_data)
        self._dimension = training_data.shape[1] - 1

    def _compute_main_and_covariance_matrix(self, training_data):
        # Compute parameters and covariance matrix.
        self.mean_ = training_data.groupby(self.class_column_name).mean().as_matrix()
        self.cov_ = []
        grouped_cov = training_data.groupby(self.class_column_name).cov()
        for c in self._class_list:
            self.cov_.append(grouped_cov.loc[c].as_matrix())

    def likelihood(self, x, class_):
        class_idx = self._class_list.index(class_)
        log_likelihood = -(2/self._dimension * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.cov_[class_idx]))) - \
            0.5 * np.dot(
                np.dot(
                    (x - self.mean_[class_idx]).T, np.linalg.inv(self.cov_[class_idx])),
                    (x - self.mean_[class_idx])
            ) + self.prior(class_)
        return log_likelihood

    def confusion_matrix(self, predicted_class, testing_class):
        # Since multiple classes GDA doesn't have confusion matrix, so we just
        # output the predicted value and true value.
        print predicted_class
        print testing_class


class MultiDimensionsTwoClassGDA(MultiDimensionsGDA):

    def __init__(self):
        super(MultiDimensionsTwoClassGDA, self).__init__()

    def confusion_matrix(self, predicted_class, testing_class, selected_class=None):
        # selected_class defined as the value we decided as positive
        if not self.is_parameter_valid(predicted_class, testing_class):
            raise DataLengthNotMatchError
        # Compute each variable as 0
        if not selected_class:
            selected_class = self._class_list[0]
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        n_sample = predicted_class.shape[0]
        for sample_index in range(n_sample):
            if testing_class[sample_index] == selected_class:
                if predicted_class[sample_index] == testing_class[sample_index]:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if predicted_class[sample_index] == testing_class[sample_index]:
                    true_negative += 1
                else:
                    false_negative += 1
        print "TP %d" %true_positive
        print "FP %d" %false_positive
        print "TN %d" %true_negative
        print "FN %d" %false_negative

        if (true_positive + false_positive) == 0:
            self.precision = np.nan
        else:
            self.precision = (true_positive) / (true_positive + false_positive)
        if (true_positive + false_positive) == 0:
            self.recall = np.nan
        else:
            self.recall = true_negative / (true_negative + false_negative)
        if self.precision == np.nan or self.recall == np.nan:
            self.f_measure = np.nan
        else:
            self.f_measure = (2*self.recall * self.precision / \
                                    (self.precision + self.recall))
        self.accuracy = (true_positive + true_negative)/ n_sample

    def perform(self):
        if not self.accuracy:
            print "Please compute confusion matrix first"
            return
        print "Accuracy = %.2f" % self.accuracy
        print "Precision = %.2f" % self.precision
        print "Recall = %.2f" % self.recall
        print "F-measure = %.2f" % self.f_measure

    def roc_curve(self, test_data, iteration_count=10):
        testing_sample = test_data.as_matrix()[:, :-1]
        n_sample = test_data.shape[0]
        precision_recall = []

        for i in range(iteration_count):
            threshold = np.random.rand()
            # print "Threshold = %.2f" %threshold
            predicted_class = []
            for sample_index in range(n_sample):
                maximum_likelihood = -np.inf
                predicted = ""
                membership = []
                for idx, c in enumerate(self._class_list):
                    print (np.exp(-(
                        0.5 * np.dot(
                            np.dot(
                                (testing_sample[sample_index, :] - self.mean_[idx]).T, np.linalg.inv(self.cov_[idx])),
                                (testing_sample[sample_index, :] - self.mean_[idx])
                        )
                    )))
                    likelihood = (1/ (np.power(2*np.pi, self._dimension/2) * np.sqrt(np.linalg.det(self.cov_[idx])))) * (np.exp(-(
                        0.5 * np.dot(
                            np.dot(
                                (testing_sample[sample_index, :] - self.mean_[idx]).T, np.linalg.inv(self.cov_[idx])),
                                (testing_sample[sample_index, :] - self.mean_[idx])
                        )
                    ))) * self.prior(c)
                    membership.append(likelihood)
                # print "membership %s" %str(membership)
                # print membership[0] / membership[1]
                if membership[0] / membership[1] >= threshold:
                    predicted = self._class_list[0]
                else:
                    predicted = self._class_list[1]
                predicted_class.append(predicted)
            self.confusion_matrix(self.predict(test_data, threshold), test_data.as_matrix()[:, -1])
            precision_recall.append((self.precision, self.recall))
        p_r = np.array(precision_recall)
        p_r = np.sort(p_r, axis=0)
        plt.plot(p_r[:, 0], p_r[:, 1])
        plt.show()
        # print precision_recall

