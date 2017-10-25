from __future__ import division
from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from random import randint
#%matplotlib inline

# Lets define some use-case specific UDF(User Defined Functions)
def moving_average(data, window_size):
    """ Computes moving average using discrete linear convolution of two one dimensional sequences.
    Args:
    -----
            data (pandas.Series): independent variable
            window_size (int): rolling window size

    Returns:
    --------
            ndarray of linear convolution

    References:
    ------------
    [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def explain_anomalies(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using stationary standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies
    """
    
    avg = moving_average(y, window_size).tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    std = np.std(residual)
    return {'standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i) for
                                                       index, y_i, avg_i in zip(count(), y, avg)
              if (y_i > avg_i + (sigma*std)) | (y_i < avg_i - (sigma*std))])}


def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
    """ Helps in exploring the anamolies using rolling standard deviation
    Args:
    -----
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma (int): value for standard deviation

    Returns:
    --------
        a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # Calculate the variation in the distribution of the residual
    testing_std = pd.rolling_std(residual, window_size)
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                  testing_std_as_df.ix[window_size - 1]).round(3).iloc[:,0].tolist()
    std = np.std(residual)
    return {'stationary standard_deviation': round(std, 3),
            'anomalies_dict': collections.OrderedDict([(index, y_i)
                                                       for index, y_i, avg_i, rs_i in zip(count(),
                                                                                           y, avg_list, rolling_std)
              if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])}

def outlier_detector(x, y, window_size, sigma_value=1, depict=False, applying_rolling_std=False):
    """ Helps in generating the plot and flagging the anamolies.
        Supports both moving and stationary standard deviation. Use the 'applying_rolling_std' to switch
        between the two.
    Args:
    -----
        x (pandas.Series): dependent variable
        y (pandas.Series): independent variable
        window_size (int): rolling window size
        sigma_value (int): value for standard deviation
        depict (boolean) :  True/False for visualizing or not
        applying_rolling_std (boolean): True/False for using rolling vs stationary standard deviation
    """
    # Query for the anomalies and plot the same
    events = {}
    if applying_rolling_std:
        events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
    else:
        events = explain_anomalies(y, window_size=window_size, sigma=sigma_value) 
    if depict:
        plt.figure(figsize=(15, 8))
        plt.plot(x, y, "k.")
        y_av = moving_average(y, window_size)
        plt.plot(x, y_av, color='green')
        
        x_anomaly = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
        y_anomaly = np.fromiter(events['anomalies_dict'].values(), dtype=float,
                                                count=len(events['anomalies_dict']))
        x_anomaly = x[x_anomaly]
        plt.plot(x_anomaly, y_anomaly, "r*", markersize=12)

        # add grid and lines and enable the plot
        plt.grid(True)
        plt.show()
    return events

if __name__=="__main__":
    # 1. Download sunspot dataset and upload the same to dataset directory
    #    Load the sunspot dataset as an Array
    #!mkdir -p dataset
    #!wget -c -b http://www-personal.umich.edu/~mejn/cp/data/sunspots.txt -P dataset
    data = loadtxt("data/sunspots.txt", float)

    # 2. View the data as a table
    data_as_frame = pd.DataFrame(data, columns=['Months', 'SunSpots'])
    data_as_frame.head()
    # 4. Lets play with the functions
    x = data_as_frame['Months']
    Y = data_as_frame['SunSpots']

    # plot the results
    outliers = outlier_detector(x, y=Y, window_size=10, sigma_value=3, depict=True)
    events = explain_anomalies(Y, window_size=5, sigma=3)
    # Display the anomaly dict
    print("Information about the anomalies model:{}".format(events))

