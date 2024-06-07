from calendar import c
from numba import jit
import numpy as np
from pyparsing import col


@jit(nopython=True)
def calculate_squared_errors(y1, y2):
    squared_errors = np.zeros(len(y1))
    for i in range(len(y1)):
        squared_errors[i] = (y1[i] - y2[i]) ** 2
    return squared_errors


def calculate_sum_root_mean_squared_errors(interpolated_array_, simulated_array_):
    sum_root_mean_squared_errors = 0.0
    for column in interpolated_array_.columns:
        y1 = interpolated_array_[column]
        y2 = simulated_array_[column]
        n = len(y1)
        sum_squared_error = 0.0
        for j in range(n):
            sum_squared_error += (y1[j] - y2[j]) ** 2
        root_mean_squared_errors = np.sqrt(sum_squared_error / n)
        sum_root_mean_squared_errors += root_mean_squared_errors
    return sum_root_mean_squared_errors
