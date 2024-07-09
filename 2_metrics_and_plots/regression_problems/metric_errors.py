'''
The most common metric in regression is error. Error is simple and very easy to 
understand.

Error = True - Predicted
Absolute error = |Error|

'''

import numpy as np
def mean_absolute_error(y_true, y_pred):
    """
    This function calculates mae
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate absolute error 
        # and add to error
        error += np.abs(yt - yp)
    # return mean error
    return error / len(y_true)

def mean_squared_error(y_true, y_pred):
    """
    This function calculates mse
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared error 
        # and add to error
        error += (yt - yp) ** 2
    # return mean error
    return error / len(y_true)

'''
Another type of error in same class is squared logarithmic error. Some people call it SLE
MSLE (mean squared logarithmic error)
'''
def mean_squared_log_error(y_true, y_pred):
    """
    This function calculates msle
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared logarithmic error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared log error 
        # and add to error
        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2
    # return mean error
    return error / len(y_true)

'''
Root mean squared logarithmic error is just a square root of this. It is also known 
as RMSLE. 
'''
def mean_percentage_error(y_true, y_pred):
    """
    This function calculates mpe
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate percentage error 
        # and add to error
        error += (yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)

def mean_abs_percentage_error(y_true, y_pred):
    """
    This function calculates MAPE
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate percentage error 
        # and add to error
        error += np.abs(yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)

'''
The best thing about regression is that there are only a few most popular metrics 
that can be applied to almost every regression problem. And it is much easier to 
understand when we compare it to classification metrics. 
'''

'''
coefficient of determination R^2

In simple words, R-squared says how good your model fits the data. R-squared 
closer to 1.0 says that the model fits the data quite well, whereas closer 0 means 
that model isn’t that good. R-squared can also be negative when the model just 
makes absurd predictions. 
'''

def r2(y_true, y_pred):
    """
    This function calculates r-squared score
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: r2 score
    """
    
    # calculate the mean value of true values
    mean_true_value = np.mean(y_true)
    
    # initialize numerator with 0
    numerator = 0
    # initialize denominator with 0
    denominator = 0
    
    # loop over all true and predicted values
    for yt, yp in zip(y_true, y_pred):
        # update numerator
        numerator += (yt - yp) ** 2
        # update denominator
        denominator += (yt - mean_true_value) ** 2
    # calculate the ratio
    ratio = numerator / denominator
    # return 1 - ratio
    return 1 - ratio

