#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from tools import read_data, cross_validation, plot_error
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cross_validation import train_test_split
from sklearn import linear_model


def gradient_descent(b_current, a_current, points, learning_rate):
    '''
    We assume the linear model is f(x) = a*x + b
    b_current: the value of b after ith iteration
    a_current: the value of a after ith iteration
    points: the 2-d dataset
    learning_rate: the learning rate of gradient descent.
    '''
    b_gradient = 0
    a_gradient = 0
    data_length = len(points)
    N = float(len(points))
    for i in range(0, len(points)):
        b_gradient += - \
            (2/N) * (points[i][1] - ((a_current*points[i][0]) + b_current))
        a_gradient += - \
            (2/N) * points[i][0] * (points[i][1] - ((a_current * points[i][0]) + b_current))
    new_b = b_current + (learning_rate * b_gradient)
    new_a = a_current + (learning_rate * a_gradient)
    error = 0
    for point_index in range(data_length):
        error += (points[point_index][1] -
                  (new_a *
                   points[point_index][0] +
                      new_b))**2
    error = error/data_length
    return [new_b, new_a, error]


def normal_equation(points, degrees=1):
    '''
    X*beta = Y
    -> inv(X.T * X) * X.T * Y = beta
    The default value of degree is 1
    '''
    # one_vector is a vector with size (len(data), 1) and all values are 1.
    one_vector = np.ones((len(points), 1), dtype=np.float)
    X_origin = np.matrix(points[:, :-1])
    X = np.matrix(points[:, :-1])
    for degree in range(2, degrees+1):
        X = np.column_stack((X, np.power(X_origin, degree)))
    X = np.column_stack((one_vector, X))
    Y = np.matrix(points[:, -1])
    Y = Y.T
    inv_Xt_X = inv((X.T)*X)
    beta = inv_Xt_X * X.T * Y
    return beta


def create_polynomial_regression_function(x, parameters):
    # Initialize function
    y = 0
    if parameters.shape[0] < 1:
        return "Parameter Invalid"
    degree = 0
    for coefficient_index in range(parameters.shape[0]):
        y += parameters.item((coefficient_index, 0)) * np.power(x, degree)
        degree += 1
    return y


def draw_trisurf(np_array):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(
        np_array[:, 0],
        np_array[:, 1],
        np_array[:, 2],
        cmap=cm.jet,
        linewidth=0.2
    )
    plt.show()


def gradient_descent_model(data, interation_count=100):
    data_length = len(data)
    parameters_and_error = []
    current_b, current_a, error = gradient_descent(0, 1, data, 0.01)
    parameters_and_error.append([current_a, current_b, error])
    for i in range(1, 100):
        new_b, new_a, error = gradient_descent(
            current_b, current_a, data, 0.0001)
        parameters_and_error.append([new_a, new_b, error])
        current_a = new_a
        current_b = new_b
    return np.array([[new_b], [new_a]])


def compute_error(data, a, b):
    '''
    Computer RSS
    '''
    error = 0
    for index in range(len(data)):
        row = data[index]
        x = row[0]
        y1 = a*x+b
        y = row[1]
        error += (y-y1)**2
    return error


def model_test(data, test_count=10):
    '''
    Using 10-viladition-cross
    '''
    training_error = []
    testing_error = []
    for i in range(test_count):
        print "############## Round %s #####################" % str(i+1)
        X_train, X_test, y_train, y_test = train_test_split(
            data[:, 0], data[:, 1], test_size=0.1, random_state=i
        )
        train_data = np.column_stack((X_train, y_train))
        test_data = np.column_stack((X_test, y_test))
        train_result = gradient_descent_model(train_data)
        training_error.append(
            compute_error(
                train_data,
                train_result[1][0],
                train_result[0][0]
            )
        )
        testing_error.append(
            compute_error(
                test_data,
                train_result[1][0],
                train_result[0][0]
            )
        )
        print "############## End %s #####################" % str(i+1)
    print "train error"
    print training_error
    print "testing error"
    print testing_error


def given_linear_regression(datafile):
    '''
    Compare performances of my function with the given regression function.
    '''

    # Import data
    data = read_data(datafile)
    col, row = data.shape

    testing_data, testing_size, training_data, training_size = cross_validation(
        data, 1)

    # Define the range of x
    x_max = int(max(data[:, 0]))+2
    x_min = int(min(data[:, 0]))-2
    x = np.array([i for i in range(x_min, x_max)])
    x_length = len(x)
    one_vector_ = np.ones((x_length, 1), dtype=np.float)
    X = np.column_stack((one_vector_, x.reshape(x_length, 1)))

    # Draw my function's fitting result graph.
    # plt.subplot(2, 1, 1)
    # plot_regression_model(
        # "My Method",
        # training_data,
        # training_size,
        # testing_data,
        # testing_size,
        # 1)

    # Draw given function's fitting result graph
    one_vector = np.ones((training_size, 1), dtype=np.float)
    one_vector_test = np.ones((testing_size, 1), dtype=np.float)
    Z = np.column_stack((one_vector, training_data[:, 0]))
    Z_test = np.column_stack((one_vector_test, testing_data[:, 0]))
    # print Z
    clf = linear_model.LinearRegression()
    clf.fit(Z, training_data[:, 1])
    print("Residual sum of squares: %.2f"
          % np.mean((clf.predict(Z_test) - testing_data[:, 1]) ** 2))
    y = np.dot(X, clf.coef_)
    y_training = clf.coef_ * Z
    plt.scatter(data[:, 0], data[:, 1], color="red")
    plt.plot(x, y)
    x_test = testing_data[:, 0]
    y_test = testing_data[:, 1]
    # y_predicted = clf.coef_ * x_test
    # print "Given Function:"
    # print "Trainging error: %s" % (np.dot((y_training - training_data[:, 1]).T, (y_training - training_data[:, 1]))/training_size)
    # print "Testing error: %s " % (np.dot((y_predicted - y_test).T, (y_predicted - y_test))/testing_size)

    # Show result
    # plt.show()


def compute_polynomial_regression(
        training_data,
        training_size,
        testing_data,
        testing_size,
        degree):
    '''
    Computing polynomial model from training data, the return value is
    parameter get from computataion and training and testing error.
    if 'reduce_data' parameter is set to True, then training data will be
    reduced by 25%
    '''
    degree_index = degree
    print "########## Degree %s ##########" % str(degree_index)
    parameters = normal_equation(training_data, degrees=degree)

    # Separate feature and label.
    x_training = training_data[:, 0]
    y_training = training_data[:, 1]

    # Compute parameter useing data after reducing if 'reduce_data' parameter
    # is set to True
    y_computed_by_training = create_polynomial_regression_function(
        x_training, parameters)
    # Compute testing error.
    x_test = testing_data[:, 0]
    y_test = testing_data[:, 1]
    y_computed_by_test = create_polynomial_regression_function(
        x_test,
        parameters)

    # Compute Training and testing error.
    training_error = np.dot((y_training-y_computed_by_training).T, (y_training-y_computed_by_training))/training_size
    testing_error = np.dot((y_test-y_computed_by_test).T, (y_test-y_computed_by_test))/testing_size

    # Print training error and testing error
    print "Training Error: %s" % training_error
    print "Testing Error: %s" % testing_error
    print "########## End degree %s##########" % str(degree_index)
    print "\n"
    return parameters, training_error, testing_error


def select_polynomial_model_from_cross_validation(data, degree, folder=10, reduce_training_data=False):
    '''
    Select model from k-folder cross validation. After separating dataset into
    10 parts, for each polynomial function, we compute 10 models by executing
    10 times that selecting one subset as testing data, the other for training,
    then we choose the model with minimum testing error.
    '''

    # Initialize minimum_error and optimal_parameters
    minimum_error = np.inf
    minimum_training_error = np.inf
    optimal_parameters = 0
    for testing_index in range(folder):
        training_data, training_size, testing_data, testing_size = cross_validation(
            data, testing_index)
        if reduce_training_data:
            print "Data reduced..."
            reduced_training_data_count = int(training_size * 0.75)
            training_data = np.copy(
                training_data[
                    :reduced_training_data_count,
                    :])
            training_size = reduced_training_data_count
        parameters, training_error, testing_error = compute_polynomial_regression(
            training_data, training_size, testing_data, testing_size, degree=degree)
        if testing_error < minimum_error:
            minimum_error = testing_error
            minimum_training_error = training_error
            optimal_parameters = parameters
    return optimal_parameters, minimum_error, minimum_training_error


def select_ploynomial_models(data_file, max_degree=6, save=False, reduce_training_data=False):
    '''
    Using 10-cross-validation to test polynomial model.
    '''
    data = read_data(data_file)

    # Define the range of x to draw model graph.
    if max_degree < 1:
        raise ValueError("Max_degree should be greater than 1")
    # if max_degree == 1, then this function only draws one graph, otherwise,
    # it will draw multiple graphs.
    if max_degree == 1:
        optimal_parameters, testing_error = select_polynomial_model_from_cross_validation(
            data, 1, reduce_training_data=reduce_training_data)
    else:
        optimal_parameters = 0
        minimum_testing_error = np.inf
        optimal_degree = 0

        # Collect the smallest errors for each degree for plotting
        # Degree-Error line graph.
        error_collections = {}
        training_error_collections = {}
        for i in range(1, max_degree+1):
            parameters, testing_error, training_error = select_polynomial_model_from_cross_validation(
                data, i, reduce_training_data=reduce_training_data
            )

            # Save degree and error in a dict.
            error_collections[i] = testing_error
            training_error_collections[i] = training_error
            if testing_error < minimum_testing_error:
                optimal_degree = i
                minimum_testing_error = testing_error
                optimal_parameters = parameters
    if save:
        plt.savefig("polynomial_model_graph.png")
    # return optimal_parameters, optimal_degree
    plt.scatter(data[:, 0], data[:, 1], color="red")
    x_max = int(max(data[:, 0]))+2
    x_min = int(min(data[:, 0]))-2
    x = np.array([i for i in range(x_min, x_max)])
    y = create_polynomial_regression_function(x, optimal_parameters)
    plt.plot(x, y)
    print "Optimal Degree %s" %str(optimal_degree)
    print "Optimal Paremeters %s" %str(optimal_parameters)
    return error_collections, training_error_collections
    # plt.show()


def draw_scatter_graph(files, save=False):
    '''
    Plot scatter graph for each dataset. If save is True, then this function will
    save result graph as "data_scatter_graph.png"
    '''
    graph_index = 1
    for f in files:
        data = read_data(f)
        # draw sub graph by graph index.
        plt.subplot(2, 2, graph_index)
        plt.title(f)
        plt.scatter(data[:, 0], data[:, 1], color="red")
        graph_index += 1
    if save:
        plt.savefig("data_scatter_plot.png")
    plt.show()


def select_linear_model(datafile, reduce_training_data=False):
    data = read_data(datafile)
    one_vector = np.ones((len(data), 1), dtype=np.float)
    Z = np.column_stack((one_vector, data))
    k = 10
    plt.scatter(data[:, 0], data[:, 1], color="red")
    minimum_error = np.inf
    optimal_parameter = 0
    for testing_index in range(k):
        testing_data, testing_size, training_data, training_size = cross_validation(
            data, testing_index)
        print "Training size = %s" %str(training_size)
        if reduce_training_data:
            print "Data reduced..."
            reduced_training_data_count = int(training_size * 0.75)
            training_size = reduced_training_data_count
            print "Reduced training data size = %s" %str(training_size)
            training_data = np.copy(
                training_data[
                    :reduced_training_data_count,
                    :])
        parameters = normal_equation(training_data)
        predicted_by_training_data = create_polynomial_regression_function(
            training_data[:, 0],
            parameters)
        training_error = np.dot(
            (predicted_by_training_data - training_data[:, 0]).T,
            (predicted_by_training_data - training_data[:, 0] )) / training_size
        training_error = training_error/training_size
        testing_data_x = testing_data[:, 0]
        testing_data_y = testing_data[:, 1]
        predicted_testing_data = create_polynomial_regression_function(
            testing_data_x,
            parameters)
        testing_error = np.dot(
            (predicted_testing_data - testing_data_y).T,
            (predicted_testing_data - testing_data_y)) / testing_size
        if testing_error < minimum_error:
            minimum_error = testing_error
            optimal_parameter = parameters
        print "Training Error %s" % training_error
        print "Testing Error %s" % testing_error
        print "Parameter %s " % parameters
        print "\n"
    x_max = int(max(data[:, 0]))+2
    x_min = int(min(data[:, 0]))-2
    x = np.array([i for i in range(x_min, x_max)])
    y = create_polynomial_regression_function(x, optimal_parameter)
    plt.plot(x, y)
    print "The minimum testing error we've got from this training is: \n %s" % minimum_error
    print "Model parameters got from this training data is: \n %s" % optimal_parameter
    print "\n"
    # plt.show()


if __name__ == "__main__":
    files = ["svar-set1.dat", "svar-set2.dat", "svar-set3.dat", "svar-set4.dat"]
    # select_linear_model("svar-set1.dat", reduce_training_data=True)
    # print select_ploynomial_models("mvar-set1.dat", reduce_training_data=True)
    # data = read_data("svar-set1.dat")
    # print gradient_descent_model(data)


    ############################################################
    # Plot linear model for each dataset
    # plot_index = 1
    # for datafile in files:
        # print "Analysis for %s" % datafile
        # plt.subplot(2,2, plot_index)
        # plt.title(datafile)
        # select_linear_model(datafile, reduce_training_data=True)
        # plot_index += 1
    # plt.show()
    ############################################################


    ############################################################
    # Plot linear regression model using given linear function
    # plot_index = 1
    # for datafile in files:
        # plt.subplot(2,2, plot_index)
        # plt.title(datafile)
        # given_linear_regression(datafile)
        # plot_index += 1
    # plt.show()
    ############################################################


    ############################################################
    # Select and plot polynomial regression model on the top of
    # data
    plot_index = 1
    errors = []
    training_errors = []
    for datafile in files:
        plt.subplot(2,2, plot_index)
        plt.title(datafile)
        testing_error, training_error = select_ploynomial_models(datafile)
        # testing_error, training_error = select_ploynomial_models(datafile, reduce_training_data=True)
        errors.append(testing_error)
        training_errors.append(training_error)
        plot_index += 1
    print errors
    plt.show()
    # plot_error(errors, training_errors)

    ############################################################



    ############################################################
    # select_linear_model("svar-set1.dat", reduce_training_data=True)
    # draw_scatter_graph(files, save=True)
    # select_ploynomial_models("svar-set1.dat")
    # error_data = gradient_descent_model(data, 1000)
    # parameters = normal_equation(data)

    # print error_data[-1]
    # print "##########"
    # print model_test(data, 10)
    # print "##########"
    # draw_trisurf(np.array(error_data))
    # plt.scatter(data[:, 0], data[:, 1], color="red")
    # plt.savefig("svar-set4.png")
    # a = error_data[1][0]
    # b = error_data[0][0]
    # x_max = int(max(data[:, 0]))+10
    # x = np.array([i for i in range(x_max)])
    # y = create_polynomial_regression_function(x, parameters)
    # plt.plot(x, y)
    # plt.show()
    # y = a*x+b
    # print "a=%s, b=%s" % (a, b)
    # y = create_polynomial_regression_function(x, parameters)

    # compare_with_given_regression()
    # Computing training and testing error
    # testing_data, testing_size, training_data, training_size = cross_validation(data)
    # print "Training Error: %s" % (compute_error(training_data, a, b))
    # print "Testing Error: %s" % (compute_error(testing_data, a, b))

    # print normal_equation(data, degrees=3)
    # print normal_equation(data)
