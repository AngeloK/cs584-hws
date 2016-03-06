#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import binom


class BasicNaiveBayes(object):

    def __init__(self):
        pass

    def _split_class_and_compute_parmeters(self, y):
        from collections import Counter
        # Compute of the number each class apppearing in training data set using
        # buildin python tool
        self.n_sample = len(y)
        class_count = Counter(y)
        self._class_list = [k for k in class_count.keys()]
        self.priors = {}
        for label in self._class_list:
            self.priors[label] = [class_count[label]/self.n_sample]
        self.priors = pd.DataFrame(self.priors)

class TwoClassBinaryFeatureNB(BasicNaiveBayes):

    def __init__(self):
        super(TwoClassBinaryFeatureNB, self).__init__()

    def _compute_probability_of_each_word(self, training_data, word, class_):
        return training_data[(training_data[word] == 1) & (training_data["class"] == class_)].shape[0]/self.n_sample

    def train(self, training_data):
        self.words_vector = training_data.columns[:-1]
        self._split_class_and_compute_parmeters(training_data.as_matrix()[:, -1])
        self.probability_table = []
        self.probability_vector = {}
        for class_ in self._class_list:
            for word in training_data.columns:
                self.probability_vector[word] = \
                    [self._compute_probability_of_each_word(
                        training_data,
                        word,
                        class_=class_
                    )]
            self.probability_table.append(pd.DataFrame(self.probability_vector))
        self.probability_table = pd.concat(self.probability_table, keys=self._class_list)

    def classify(self, test_word_vector):
        '''
        The data type for test_word_vector is pandas DataFrame:

            #  word1  word2  word3
            0    1      0      1

        '''
        max_likelihood = -np.inf
        predicted_class = None
        for c in self._class_list:
            likelihood = 0
            for each_word in self.words_vector:
                likelihood += self.log_bernoulli_function(
                    self.probability_table.ix[int(c)][each_word][0],
                    test_word_vector[each_word]
                )
            likelihood += self.priors[c][0]
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                predicted_class = c
        return predicted_class

    def log_bernoulli_function(self, alpha, x):
        a = x * np.log(alpha) + (1 - x) * np.log(1 - alpha)
        return a

    def predict(self, test_data):
        n_test_sample = test_data.shape[0]
        predicted_list = []
        for idx in test_data.index:
            test_sample = test_data.ix[idx]
            predicted_list.append(
                self.classify(test_sample)
            )
        return predicted_list

    def perform(self, predicted, test):
        count = 0
        for i in range(len(predicted)):
            if predicted[i] == test[i]:
                count += 1
        print count


class TwoClassDiscreteFeatureNB(BasicNaiveBayes):

    def __init__(self):
        super(TwoClassDiscreteFeatureNB, self).__init__()

    def build_dictionary(self, word_count_set, minimum_count):
        self.dictionary = {}
        for key, val in word_count_set.iteritems():
            if val >= minimum_count:
                self.dictionary[key] = [val]
        self.dictionary = pd.DataFrame(self.dictionary)

    def train(self, training_word_matrix, label):
        self.n_sample = label.shape[0]
        word_count_set = Counter(training_word_matrix["word_index"])
        self.build_dictionary(word_count_set, 0)
        self.compute_priors(label)
        self.compute_alpha(training_word_matrix, label)

    def compute_priors(self, label):
        class_count = Counter(label["class"])
        self._class_list = [k for k in class_count.keys()]
        priors = {}
        for c in self._class_list:
            priors[c] = [
                label[label["class"]== c].count()[0]/self.n_sample
            ]
        self.priors = pd.DataFrame(priors)

    def compute_alpha(self, training_data, label):
        alpha_table = []
        for c in self._class_list:
            alpha = {}
            idx = label[label["class"] == c].index
            # Change to 1-index
            idx += 1
            data_idx_with_given_class = idx
            samples = training_data[training_data.email_id.isin(data_idx_with_given_class)]
            word_subset = samples[samples.word_index.isin(self.dictionary.columns)]
            total_word_count = word_subset["count"].sum()
            for word in self.dictionary.columns:
                current_word_count = samples[samples.word_index == word]["count"].sum()
                alpha[word] = (current_word_count + 1)/(total_word_count + self.dictionary.shape[1])
            alpha_table.append(alpha)
        self.alpha_table = pd.DataFrame(alpha_table)

    def log_bionimial_function(self, word_vector, class_):
        f = 0
        for x in word_vector["word_index"]:
            if x in self.dictionary:
                f += binom.pmf(word_vector[word_vector["word_index"] == x]["count"].sum(),
                                 word_vector["count"].sum(),
                                 self.alpha_table.ix[class_][x]
                               )
        return f

    def classify(self, word_vectors):
        # TODO define the data structure of word_vectors
        predicted_list = {}
        maximal = -np.inf
        predicted_class = None
        email_ids = []
        for email_id, each_vector in word_vectors.groupby("email_id"):
            email_ids.append(email_id)
            for c in self._class_list:
                likelihood = self.log_bionimial_function(each_vector, c)
                if likelihood > maximal:
                    maximal = likelihood
                    predicted_class = c
            predicted_list[email_id] = [predicted_class]
        return pd.DataFrame(predicted_list, index=["class"]).T

    def perform(self, predicted_list, test_label):
        # TODO: evaluate the accuracy
        count = 0
        for idx in predicted_list.index:
            if predicted_list.ix[idx][0] == test_label.ix[idx][0]:
                count += 1
        return count
