
from cmath import sqrt
import numpy as np
import time
import sys
import cupy as cp
import psutil
import os
from numba import jit, njit, vectorize, prange
import concurrent.futures
import shutil
import pandas as pd

GOLDEN_RATIO = ((1+sqrt(5))/2).real


def no_consecutive_ones(length):
    end_in_zero = cp.zeros((1, 1), dtype=int)
    end_in_one = cp.ones((1, 1), dtype=int)

    for i in range(length-1):
        end_in_zero_zero = cp.append(end_in_zero,
                                     cp.zeros(
                                         (end_in_zero.shape[0], 1), dtype=int),
                                     axis=1)
        end_in_zero_one = cp.append(end_in_zero,
                                    cp.ones(
                                        (end_in_zero.shape[0], 1), dtype=int),
                                    axis=1)
        end_in_one_zero = cp.append(end_in_one,
                                    cp.zeros(
                                        (end_in_one.shape[0], 1), dtype=int),
                                    axis=1)
        end_in_zero = cp.append(end_in_zero_zero, end_in_one_zero, axis=0)
        end_in_one = end_in_zero_one

    return cp.append(end_in_one, end_in_zero, axis=0)


def no_consecutive_ones_b(length):
    end_in_zero = cp.zeros((1, 1), dtype='b')
    end_in_one = cp.ones((1, 1), dtype='b')

    for i in range(length-1):
        end_in_zero_zero = cp.append(end_in_zero,
                                     cp.zeros(
                                         (end_in_zero.shape[0], 1), dtype='b'),
                                     axis=1)
        end_in_zero_one = cp.append(end_in_zero,
                                    cp.ones(
                                        (end_in_zero.shape[0], 1), dtype='b'),
                                    axis=1)
        end_in_one_zero = cp.append(end_in_one,
                                    cp.zeros(
                                        (end_in_one.shape[0], 1), dtype='b'),
                                    axis=1)
        end_in_zero = cp.append(end_in_zero_zero, end_in_one_zero, axis=0)
        end_in_one = end_in_zero_one

    return cp.append(end_in_one, end_in_zero, axis=0)


def golden_multiplied_2d(num_digits):
    multiplied_matrix = cp.array(
        [[GOLDEN_RATIO ** (-1), GOLDEN_RATIO ** (-num_digits)]])
    for i in range(2, num_digits+1):
        multiplied_matrix = cp.append(multiplied_matrix,
                                      [[GOLDEN_RATIO ** (-i), GOLDEN_RATIO ** (i-num_digits-1)]], axis=0)
    return multiplied_matrix


def golden_multiplied_1d(num_digits):
    multiplied_matrix = cp.array(
        [[GOLDEN_RATIO ** (-1)]])
    for i in range(2, num_digits+1):
        multiplied_matrix = cp.append(multiplied_matrix,
                                      [[GOLDEN_RATIO ** (-i)]], axis=0)
    return multiplied_matrix


def partition(num_digits):
    numbers = cp.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_1d(num_digits))
    numbers = cp.append(numbers[numbers[:, 0].argsort()], [[1]], axis=0)
    return numbers


def points(num_digits):
    points = cp.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_2d(num_digits))
    points = points[points[:, 0].argsort()]
    return points


def star_discrepancy(point_set):
    n = point_set.shape[0]

    largest_outer_value = 0

    for index_i in range(1, n-1):
        sorted = cp.argsort(point_set[:index_i, 1])

        largest_inner_value = 0

        for index_k in range(index_i-1):
            if largest_inner_value < max(index_k / n - point_set[index_i, 0] * point_set[sorted[index_k], 1],                                    point_set[index_i+1, 0]*point_set[sorted[index_k+1], 1]-index_k / n):
                largest_inner_value = max(index_k / n - point_set[index_i, 0] * point_set[sorted[index_k], 1],
                                          point_set[index_i+1, 0]*point_set[sorted[index_k+1], 1]-index_k / n)
        index_k = index_i
        if largest_inner_value < max(index_i / n - point_set[index_i, 0] * point_set[sorted[-1], 1], point_set[index_i+1, 0]-index_i / n):
            largest_inner_value = max(
                index_i / n - point_set[index_i, 0] * point_set[sorted[-1], 1], point_set[index_i+1, 0]-index_i / n)

        if largest_outer_value < largest_inner_value:
            largest_outer_value = largest_inner_value

    index_i = n-1
    sorted = cp.argsort(point_set[:index_i, 1])
    largest_inner_value = 0

    for index_k in range(index_i-1):
        if largest_inner_value < max(index_k / n - point_set[index_i, 0] * point_set[sorted[index_k], 1], point_set[sorted[index_k+1], 1]-index_k / n):
            largest_inner_value = max(
                index_k / n - point_set[index_i, 0] * point_set[sorted[index_k], 1], point_set[sorted[index_k+1], 1]-index_k / n)

    if largest_inner_value < 1 - point_set[-1, 0] * point_set[sorted[-1], 1]:
        largest_inner_value = 1 - point_set[-1, 0] * point_set[sorted[-1], 1]

    if largest_outer_value < largest_inner_value:
        largest_outer_value = largest_inner_value

    return largest_outer_value


start_time = time.perf_counter()
num_digits = 10
point_set = points(num_digits)
print(star_discrepancy(point_set))
end_time = time.perf_counter()
time_difference = end_time-start_time
print(f"Star discrepancy time : {time_difference :0.6f}")
