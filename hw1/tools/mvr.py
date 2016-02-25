#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from tools import read_data, cross_validation, data_split, plot_error, plot_time_cost
import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from copy import copy
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures
import time
from sklearn.metrics.pairwise import rbf_kernel


def create_polynomial_feature_matrix(data_x, degree):
    polynomial_feature_matrx = PolynomialFeatures(degree = degree)
    return polynomial_feature_matrx.fit_transform(data_x)


def mvar_regression(testing_data, testing_size, training_data, training_size, degree=5):
    one_vector = np.ones((len(training_data), 1), dtype=np.float)
    X = create_polynomial_feature_matrix(training_data[:, :-1], degree=degree)
    print "X's shape: %s" %str(X.shape)
    theta = np.dot(pinv(X), training_data[:, -1])
    print "theta shape %s" %str(theta.shape)
    y_predicted_by_model = np.dot(X, theta.T).T
    X_test = create_polynomial_feature_matrix(testing_data[:, :-1], degree=degree)
    y_test = testing_data[:, -1]
    y_test_predicted = np.dot(X_test, theta.T).T
    training_error = np.dot((y_predicted_by_model - training_data[:, -1]), (y_predicted_by_model - training_data[:, -1]).T)/training_size
    testing_error = np.dot((y_test_predicted - y_test), (y_test_predicted - y_test).T)/testing_size
    print "##### Max Degree %s #####" % int(degree)
    print "Training Error: %s " % training_error
    print "Testing Error: %s" % testing_error
    print "##### Max Degree %s End #####" % int(degree)
    print "\n"
    return theta, training_error, testing_error


def select_mvar_model_from_cross_validation(datafile, max_degree, folder=10):
    data = read_data(datafile)
    minimum_error = np.inf
    minimum_training_error = np.inf
    global_mimimum_error = np.inf
    optimal_theta = 0
    global_optimal_theta = 0
    global_optimal_degree = 0
    testing_error_collection = {}
    training_error_collection = {}
    time_collection = {}
    for degree_index in range(1, max_degree+1):
        time_start = time.time()
        for testing_data_index in range(folder):
            testing_data, testing_size, training_data, training_size = cross_validation(data, testing_data_index)
            theta, training_error, testing_error = mvar_regression(testing_data, testing_size, training_data, training_size, degree_index)
            if testing_error < minimum_error:
                minimum_error = testing_error
                minimum_training_error = training_error
                optimal_theta = theta
        print "Min error: %s" %str(minimum_error)
        testing_error_collection[degree_index] = minimum_error
        training_error_collection[degree_index] = minimum_training_error
        if minimum_error < global_mimimum_error:
            global_mimimum_error = minimum_error
            global_optimal_theta = optimal_theta
            global_optimal_degree = degree_index
        time_end = time.time()
        time_cost = time_end - time_start
        time_collection[degree_index] = time_cost
    print "Global Optimal Theta: %s" % global_optimal_theta
    print "Global Optimal Degree: %s" % global_optimal_degree
    return testing_error_collection, training_error_collection, time_collection


def gradient_descent(data, parameters, degree, learning_rate=0.1):
    feature_matrix = data[:, :-1]
    label_matrix = data[:, -1]
    polynomial_feature_matrix = create_polynomial_feature_matrix(feature_matrix, degree)
    col, row = polynomial_feature_matrix.shape
    print label_matrix
    print "col= %s" % str(col)
    # parameters = np.array([1 for _ in range(row)], dtype = np.float)
    temp_parameters_after_computing = np.copy(parameters)
    for parameter_index in range(len(parameters)):
        temp_parameters = np.copy(parameters)
        # print "temp_parameters = %s" %str(temp_parameters)
        for sample_index in range(col):
            print "Poly_matrix =%s" %str(polynomial_feature_matrix[sample_index, :])
            y = polynomial_funcion(polynomial_feature_matrix[sample_index, :], parameters)
            print "Y=%s" %str(y)
            print "cofficient = %s" % str(polynomial_feature_matrix[sample_index][parameter_index])
            print "labe_y = %s" %str(label_matrix[sample_index])
            temp_parameters[parameter_index] += \
                (2/col) * polynomial_feature_matrix[sample_index][parameter_index] * (y - label_matrix[sample_index])
            print "temp_parameters in loop = %s" %str(temp_parameters)
        temp_parameters_after_computing[parameter_index] = temp_parameters[parameter_index]
        print "temp_parameters_after_computing = %s" % str(temp_parameters_after_computing)
        # print temp_parameters
    parameters = parameters - (learning_rate * temp_parameters_after_computing)
    print "parameter=%s" % str(parameters)
    return parameters


def polynomial_funcion(X, parameters):
    y = np.dot(X, parameters.T)
    return y


def stochastic_gradient_descent(polynomial_feature_matrix, label_matrix, parameters, degree, learning_rate):
    sample_count, dimension = polynomial_feature_matrix.shape
    temp_parameters_after_computing = np.copy(parameters)
    for parameter_index in range(len(parameters)):
        temp_parameters = np.copy(parameters)
        sample_index = np.random.randint(0, sample_count)
        y = polynomial_funcion(polynomial_feature_matrix[sample_index, :], parameters)
        temp_parameters[parameter_index] = \
            polynomial_feature_matrix[sample_index][parameter_index] * (y - label_matrix[sample_index])
        temp_parameters_after_computing[parameter_index] = temp_parameters[parameter_index]
    parameters = parameters - (learning_rate * temp_parameters_after_computing)
    return parameters


def iterative_compute_gd(data, degree, interation_count=100):
    feature_matrix = data[:, :-1]
    label_matrix = data[:, -1]
    polynomial_feature_matrix = create_polynomial_feature_matrix(feature_matrix, degree)
    parameters_length = polynomial_feature_matrix.shape[1]
    parameters = np.array([1 for _ in range(parameters_length)], dtype = np.float)
    parameters = stochastic_gradient_descent(polynomial_feature_matrix, label_matrix, parameters, degree, 0.1)
    for iteration_index in range(1, interation_count):
        parameters = stochastic_gradient_descent(polynomial_feature_matrix, label_matrix, parameters, degree, 0.1)
    return parameters


def gaussian_kernel_function(x_i, x, sigma):
    '''
    Define Gaussian kernel function
    '''
    # Compute Euclidean distance between two vectors.
    distance = np.sqrt(sum((x_i - x) ** 2))
    return np.exp(- distance /(2 * sigma * sigma))


def gram_matrix(x):
    '''
    Compute gram matrix by feature matrix x
    G = np.dot(X, X.T)
    '''
    return np.dot(x, x.T)


def compute_alpha(gram_matrix, label_vector, lambda_):
    col, row = gram_matrix.shape
    I = np.identity(col)
    alpha = np.dot(inv(gram_matrix + lambda_ * I), label_vector)
    return alpha


def dual_regression(training_data, testing_data, lambda_, sigma):
    # compute Gram Matrix
    G = gram_matrix(training_data[:, :-1])
    alpha = compute_alpha(G, training_data[:, -1], lambda_)
    y_predicted = predicted_value_of_dual_regression(alpha, training_data[:, :-1], testing_data[:, :-1], sigma)
    testing_error = compute_testing_error(y_predicted, testing_data[:, -1])
    return testing_error


def predicted_value_of_dual_regression(alpha, training_data_x, testing_data_x, sigma):
    '''
    Compute predicted values by alpha and testing feature matrix.
    alpha is dual linear regression coefficients.
    sigam is ussd for computing Gaussian kernel matrix.
    '''
    training_col, training_row = testing_data_x.shape
    testing_col, testing_row = testing_data_x.shape
    K = rbf_kernel(training_data_x, testing_data_x)
    return np.dot(alpha, K)


def compute_testing_error(predicted_value, testing_value):
    testing_data_size = testing_value.shape[0]
    return sum((predicted_value - testing_value) ** 2) / testing_data_size


def analyze_stochastic_gradient_descent(datafile):
    '''
    Compute
    '''
    time_start = time.time()
    minimum_error = np.inf
    optimal_parameter = 0
    data = read_data(datafile)
    for testing_data_index in range(10):
        testing_data, testing_size, training_data, training_size = cross_validation(data, testing_data_index)
        parameters = iterative_compute_gd(training_data, 2)
        y = predicted_value_of_dual_regression(parameters, testing_data[:, :-1], 2)
        testing_error = compute_testing_error(y, testing_data[:, -1])
        if testing_error < minimum_error:
            minimum_error = testing_error
            optimal_parameter = parameters
    time_end = time.time()
    time_cost = time_end - time_start
    print time_cost, minimum_error, optimal_parameter


def analyze_dual_regression(datafile, lambda_, sigma):
    data = read_data(datafile)
    minimum_error = np.inf
    time_cost = 0
    time_start = time.time()
    for folder_index in range(10):
        testing_data, testing_size, training_data, training_size = cross_validation(data, folder_index)
        testing_error = dual_regression(training_data, testing_data, lambda_ = 0.5, sigma=0.5)
        if testing_error < minimum_error:
            minimum_error = testing_error
    time_end = time.time()
    print "time_costing %s" % str(time_end - time_start)
    print "Minimum Testing Error %s" %str(testing_error)


if __name__ == "__main__":
    datafiles = ["mvar-set1.dat", "mvar-set2.dat", "mvar-set3.dat", "mvar-set4.dat"]
    testing_errors = []
    training_errors = []
    time_collection = []
    ################################################################################
    # Primal Linear Regression
    ################################################################################

    # for datafile in datafiles:
        # testing_error, training_error, time_cost = select_mvar_model_from_cross_validation(datafile, 1)
        # print time_cost
        # print testing_error
        # testing_errors.append(testing_error)
        # training_errors.append(training_error)
        # time_collection.append(time_cost)
    # plot_error(testing_errors, training_errors)
    # plot_time_cost(time_collection)

    ################################################################################
    # Primal Linear Regression End
    ################################################################################

    ################################################################################
    # Gradient Descent
    ################################################################################

    # stochastic_gradient_descent("mvar-set4.dat", 2)
    for datafile in datafiles:
        analyze_stochastic_gradient_descent(datafile)


    ################################################################################
    # Gradient Descent End
    ################################################################################


    ################################################################################
    # Dual Linear Regression
    ################################################################################
    # for datafile in datafiles:
        # analyze_dual_regression(datafile, 0.5, 0.5)

    ################################################################################
    # Dual Linear Regression End
    ################################################################################
