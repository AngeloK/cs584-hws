#!/usr/bin/env python
# -*- coding: utf-8 -*-


from gda import SingleDimensionTwoClassGDA, MultiDimensionsTwoClassGDA
from naive_bayes import TwoClassBinaryFeatureNB, TwoClassDiscreteFeatureNB
from evaluator import Evaluator
import numpy as np
import pandas as pd
from tools import cross_validation
from sklearn.cross_validation import KFold


if __name__ == "__main__":

    base_path = "/Users/Neyanbhbin/Documents/code/Data Analysis/cs584/hw2/dataset/"
    # sv = SingleDimensionTwoClassGDA()
    # best_sv = SingleDimensionTwoClassGDA()
    # data = pd.read_csv(base_path+"iris/2-d-iris.csv")
    # best_accuracy = 0
    kf = KFold(100, n_folds=10, shuffle=True)
    # for train_index, test_index in kf:
        # print "===="
        # train_data, test_data = data.ix[train_index], data.ix[test_index]
        # sv.train(train_data, -1)
        # t_label = test_data.as_matrix()[:, -1]
        # p_label = sv.predict(test_data)
        # print "Testing Label %s" %str(t_label)
        # print "Predicted Label %s" %str(p_label)
        # sv.confusion_matrix(p_label, t_label)
        # sv.perform()
        # if sv.accuracy > best_accuracy:
            # best_sv = sv
        # print "===="
        # print "\n"
    # t_whole_label = data.as_matrix()[:, -1]
    # best_sv.confusion_matrix(best_sv.predict(data), t_whole_label)
    # print best_sv.predict(data)
    # best_sv.confusion_matrix(data, data.as_matrix()[:, -1])
    # best_sv.perform()

    # data1 = pd.read_csv(base_path+"iris/iris.csv")
    # best_mv = MultiDimensionsTwoClassGDA()
    # best_acc = 0
    # for train_index, test_index in kf:
        # print "===="
        # train_data, test_data = data1.ix[train_index], data1.ix[test_index]
        # mg = MultiDimensionsTwoClassGDA()
        # mg.train(train_data, -1)
        # mg.confusion_matrix(mg.predict(test_data), test_data.as_matrix()[:, -1])
        # if mg.accuracy > best_acc:
            # best_acc = mg.accuracy
            # best_mv = mg
        # mg.perform()
        # print "===="
        # print "\n"
    # p_ = best_mv.predict(data1)
    # best_mv.confusion_matrix(p_, data1.as_matrix()[:, -1])


    # print mg.mean_
    # print mg.cov_
    # print mg.predict(test_data)

    # data = pd.read_csv(base_path+"spambase/spambase.csv")
    # kf = KFold(data.shape[0], n_folds=10, shuffle=True)
    # nb = TwoDimensionNB()
    # for train_index, test_index in kf:
        # train_data, test_data = data.ix[train_index], data.ix[test_index]
    # nb.train(train_data)
    # nb.predict(test_data)
    # nb.perform(nb.predict(test_data), test_data.as_matrix()[:, -1])

    spamdata = pd.read_csv(base_path + "ex6DataPrepared/train-features-100.csv", sep=" ")
    label = pd.read_csv(base_path + "ex6DataPrepared/train-labels-100.csv", sep=" ")
    kf = KFold(label.shape[0], n_folds=10, shuffle=True)
    best_nv = TwoClassDiscreteFeatureNB()
    max_accuracy = 0
    for train_index, test_index in kf:
        print "===="
        train_label, test_label = label.ix[train_index], label.ix[test_index]
        train_label.index += 1
        test_label.index += 1
        train_data = spamdata[spamdata["email_id"].isin(train_label.index)]
        test_data = spamdata[spamdata["email_id"].isin(test_label.index)]
        nb = TwoClassDiscreteFeatureNB()
        nb.train(train_data, train_label)
        print test_label
        print "===="
        word_vectors = test_data[test_data["email_id"].isin(test_label.index)]
        print "Predicted:\n"
        pre_label = nb.classify(word_vectors)
        print pre_label
        e = Evaluator(pre_label, test_label, 1)
        print e.score()
        if e.accuracy > max_accuracy:
            max_accuracy = e.accuracy
            best_nv = nb
    total_word_vector = spamdata[spamdata["email_id"].isin(label.index)]
    total_predict = best_nv.classify(total_word_vector)
    e_t = Evaluator(total_predict, label, 1)
    print e_t.score()
    # for email_id, df in word_vectors.groupby("email_id"):
        # print df
    # print words_vector
    # print nb.classify(word_vectors)
        # print nb.perform(nb.classify(word_vectors), test_label)
        # print "===="
    # for e in test_label.index:
        # print test_label.ix[e][0]
