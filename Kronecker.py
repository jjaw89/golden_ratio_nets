
# from os import posix_spawn
# from turtle import pos
import cupy as cp
from pathlib import Path
from cmath import sqrt
from dataclasses import replace
from msilib.schema import Binary
from tkinter import S
from xml.dom import WrongDocumentErr
import numpy as np
import time
import sys
import math
import scipy.special
from scipy.stats import qmc
from numba import jit, njit, vectorize, prange, cuda
import concurrent.futures
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import random
from sympy import GoldenRatio, plot
plt.style.use('seaborn-whitegrid')

GOLDEN_RATIO = ((1+np.sqrt(5))/2).real
LOG_GOLDEN_RATIO = math.log2(GOLDEN_RATIO).real
LOG_GOLDEN_RATIO_2 = math.log(GOLDEN_RATIO).real


def fibonacci(m):
    if m == 0:
        return 0
    elif m == 1:
        return 1
    a_i = 0
    a_i_plus = 1
    for n in range(m-1):
        a_i_plus_plus = a_i+a_i_plus
        a_i = a_i_plus
        a_i_plus = a_i_plus_plus
    return int(a_i_plus)


FIBONACCI = np.array([fibonacci(n) for n in range(64)])


def no_consecutive_ones(length):
    end_in_zero = np.zeros((1, 1), dtype='b')
    end_in_one = np.ones((1, 1), dtype='b')

    for i in range(length-1):
        end_in_zero_zero = np.append(end_in_zero, np.zeros(
            (end_in_zero.shape[0], 1), dtype='b'), axis=1)
        end_in_zero_one = np.append(end_in_zero, np.ones(
            (end_in_zero.shape[0], 1), dtype='b'), axis=1)
        end_in_one_zero = np.append(end_in_one, np.zeros(
            (end_in_one.shape[0], 1), dtype='b'), axis=1)
        end_in_zero = np.append(end_in_zero_zero, end_in_one_zero, axis=0)
        end_in_one = end_in_zero_one

    return np.append(end_in_zero, end_in_one, axis=0)


def no_consecutive_ones_reversed(length):
    end_in_zero = np.zeros((1, 1), dtype='b')
    end_in_one = np.ones((1, 1), dtype='b')

    for i in range(length-1):
        end_in_zero_zero = np.append(end_in_zero,
                                     np.zeros(
                                         (end_in_zero.shape[0], 1), dtype='b'),
                                     axis=1)
        end_in_zero_one = np.append(end_in_zero,
                                    np.ones(
                                        (end_in_zero.shape[0], 1), dtype='b'),
                                    axis=1)
        end_in_one_zero = np.append(end_in_one,
                                    np.zeros(
                                        (end_in_one.shape[0], 1), dtype='b'),
                                    axis=1)
        end_in_zero = np.append(end_in_zero_zero, end_in_one_zero, axis=0)
        end_in_one = end_in_zero_one

    return np.flip(np.append(end_in_zero, end_in_one, axis=0), axis=1)

##########################################################


def golden_multiplied_2d(num_digits):
    multiplied_matrix = np.array(
        [[GOLDEN_RATIO ** (-1), GOLDEN_RATIO ** (-num_digits)]])
    for i in range(2, num_digits+1):
        multiplied_matrix = np.append(multiplied_matrix,
                                      [[GOLDEN_RATIO ** (-i), GOLDEN_RATIO ** (i-num_digits-1)]], axis=0)
    return multiplied_matrix


def golden_multiplied_1d(num_digits):
    multiplied_matrix = np.array(
        [[GOLDEN_RATIO ** (-1)]])
    for i in range(2, num_digits+1):
        multiplied_matrix = np.append(multiplied_matrix,
                                      [[GOLDEN_RATIO ** (-i)]], axis=0)
    return multiplied_matrix


def partition(num_digits):
    if num_digits == 0:
        return np.array([0, 1]).real
    numbers = np.matmul(no_consecutive_ones(
        num_digits), golden_multiplied_1d(num_digits))
    numbers = np.append(numbers[numbers[:, 0].argsort()], [[1]], axis=0)
    return numbers[:, 0]


def hammersley_points_sorted(num_digits):
    points = np.matmul(no_consecutive_ones(
        num_digits), golden_multiplied_2d(num_digits))
    points = points[points[:, 0].argsort()]
    return points


def hammersley_points(num_digits):
    points = np.matmul(no_consecutive_ones(
        num_digits), golden_multiplied_2d(num_digits))
    points = points[points[:, 0].argsort()]
    return points


def hammersley_points_shuffled(num_digits):
    points = np.matmul(no_consecutive_ones(
        num_digits), golden_multiplied_2d(num_digits))
    np.random.shuffle(points)
    return points
##########################################################


def no_consecutive_ones_binary(length):
    numbers = np.zeros((fibonacci(length+2)), dtype='int64')
    numbers[1] = 1
    # print(numbers)
    for m in range(2, length+1):
        # print(fibonacci(m+2))
        # print(fibonacci(m+3))
        numbers[fibonacci(m+1):fibonacci(m+2)] += np.add(numbers[fibonacci(m+1)
                          :fibonacci(m+2)], numbers[:fibonacci(m)])+2**(m-1)
        # print(numbers)

    return numbers


@njit()
def no_consecutive_ones_binary_2(length):
    numbers = np.zeros((FIBONACCI[length+2]), dtype='int64')
    numbers[1] = 1
    # print(numbers)
    for m in range(2, length+1):
        # print(fibonacci(m+2))
        # print(fibonacci(m+3))
        numbers[FIBONACCI[m+1]:FIBONACCI[m+2]] += np.add(
            numbers[FIBONACCI[m+1]:FIBONACCI[m+2]], numbers[:FIBONACCI[m]])+2**(m-1)
        # print(numbers)

    return numbers


def convert_to_decimal(vectors):
    multiplied_matrix = np.array([[GOLDEN_RATIO ** (-k)]
                                  for k in range(1, vectors.shape[1]+1)])
    # print(multiplied_matrix)
    # print(np.matmul(vectors, multiplied_matrix))
    return np.matmul(vectors, multiplied_matrix)


@njit()
def integer_to_decimal(number):
    decimal = 0
    # print(number)
    for n in range(32):
        decimal += ((number // 2**n) % 2) * GOLDEN_RATIO**(-n-1)
    return decimal


@njit()
def convert_integers_to_decimals(numbers):
    return np.array([integer_to_decimal(n) for n in numbers], dtype='float')


@njit()
def convert_integer_points_to_decimals(points_n):
    return np.array([[integer_to_decimal(n[0]), integer_to_decimal(n[1])] for n in points_n], dtype='float')


# @njit(parallel=True)
# def convert_integers_to_decimals_2(numbers):
#     decimals = np.zeros((numbers.size), dtype='float')
#     for n in prange(numbers.size):
#         decimals[n] = integer_to_decimal(numbers[n])
#     return decimals


def partition_numbers_2d(digits_to_partition, x_partition_digits, y_partition_digits):
    last_digit_x = no_consecutive_ones_reversed(x_partition_digits)[:, -1]
    last_digit_y = no_consecutive_ones_reversed(y_partition_digits)[:, -1]
    return np.array([[fibonacci(digits_to_partition+2-x_partition_digits-y_partition_digits-r-c) for r in last_digit_y] for c in last_digit_x])


def partition_numbers_1d(digits_to_partition, partition_digits):
    last_digit = no_consecutive_ones_reversed(partition_digits)[:, -1]
    return np.array([fibonacci(digits_to_partition+2-partition_digits-r) for r in last_digit])


# def partition_digits(length):
#     return no_consecutive_ones_reversed(length)


@njit()
def partition_numbers_1d_binary(digits_to_partition, partition_digits):
    last_digit = np.zeros((FIBONACCI[partition_digits+2]), dtype='int')
    last_digit[1] = 1
    # print(numbers)
    for m in range(2, partition_digits+1):
        # print(fibonacci(m+2))
        # print(fibonacci(m+3))
        last_digit[FIBONACCI[m+1]:FIBONACCI[m+2]] += np.add(
            last_digit[FIBONACCI[m+1]:FIBONACCI[m+2]], last_digit[:FIBONACCI[m]])
        # print(numbers)
    return np.array([FIBONACCI[digits_to_partition+2-partition_digits-r] for r in last_digit])


@njit()
def partition_numbers_2d_binary(digits_to_partition, x_partition_digits, y_partition_digits):
    last_digit_x = np.zeros((FIBONACCI[x_partition_digits+2]), dtype='int')
    last_digit_x[1] = 1
    for m in range(2, x_partition_digits+1):
        last_digit_x[FIBONACCI[m+1]:FIBONACCI[m+2]] += np.add(
            last_digit_x[FIBONACCI[m+1]:FIBONACCI[m+2]], last_digit_x[:FIBONACCI[m]])

    last_digit_y = np.zeros((FIBONACCI[y_partition_digits+2]), dtype='int')
    last_digit_y[1] = 1
    for m in range(2, y_partition_digits+1):
        last_digit_y[FIBONACCI[m+1]:FIBONACCI[m+2]] += np.add(
            last_digit_y[FIBONACCI[m+1]:FIBONACCI[m+2]], last_digit_y[:FIBONACCI[m]])

    return np.array([[FIBONACCI[digits_to_partition+2-x_partition_digits-y_partition_digits-r-c] for r in last_digit_y] for c in last_digit_x], dtype='int')


@njit()
def no_consecutive_ones_binary_flipped(length):
    numbers = no_consecutive_ones_binary_2(length)
    # new_numbers = np.zeros_like(numbers,dtype = 'int')
    for n in range(numbers.size):
        new_number = 0
        for m in range(length):
            k = (numbers[n] // 2**m) % 2
            # print(k)
            new_number += k*2**(length-1-m)
        numbers[n] = new_number
    return numbers


def check_equidistribution(x_numbers, y_numbers, num_point_digits, num_x_partition_digits, num_y_partition_digits):
    x_partition = no_consecutive_ones_binary_flipped(num_x_partition_digits)
    y_partition = no_consecutive_ones_binary_flipped(num_y_partition_digits)
    x_look_up = np.zeros(
        (2**num_x_partition_digits), dtype='int64')
    y_look_up = np.zeros(
        (2**num_y_partition_digits), dtype='int64')

    # print(x_partition)
    # print(x_look_up)
    # print(x_partition.size)
    # print(x_look_up.size)
    for n in range(x_partition.size):
        x_look_up[x_partition[n]] = n
    # print(x_look_up)
    for n in range(y_partition.size):
        y_look_up[y_partition[n]] = n
    counting_matrix = partition_numbers_2d_binary(
        num_point_digits, num_x_partition_digits, num_y_partition_digits)

    for n in range(x_numbers.size):
        # print(
        #     f'({x_numbers[n]},{y_numbers[n]}): ({bin(x_numbers[n])},{bin(y_numbers[n])}) : ({x_numbers[n] % 2**(num_x_partition_digits)}, {y_numbers[n] % 2**(num_y_partition_digits)})')
        counting_matrix[x_look_up[x_numbers[n] % 2**(num_x_partition_digits)],
                        y_look_up[y_numbers[n] % 2**(num_y_partition_digits)]] -= 1
        # print(counting_matrix)
    return counting_matrix


@njit()
def check_equidistribution_njit(x_numbers, y_numbers, num_point_digits, num_x_partition_digits, num_y_partition_digits):
    x_partition = no_consecutive_ones_binary_flipped(num_x_partition_digits)
    y_partition = no_consecutive_ones_binary_flipped(num_y_partition_digits)
    x_look_up = np.zeros(
        (2**num_x_partition_digits), dtype='int64')
    y_look_up = np.zeros(
        (2**num_y_partition_digits), dtype='int64')

    for n in range(x_partition.size):
        x_look_up[x_partition[n]] = n
    for n in range(y_partition.size):
        y_look_up[y_partition[n]] = n

    counting_matrix = partition_numbers_2d_binary(
        num_point_digits, num_x_partition_digits, num_y_partition_digits)

    for n in range(x_numbers.size):
        counting_matrix[x_look_up[x_numbers[n] % 2**(num_x_partition_digits)],
                        y_look_up[y_numbers[n] % 2**(num_y_partition_digits)]] -= 1
    return counting_matrix


@njit()
def extend_points(point_numbers, starting_digits):
    x_numbers = point_numbers[:, 0]
    y_numbers = point_numbers[:, 1]
    current_digits = starting_digits+1
    x_numbers_to_add = no_consecutive_ones_binary_2(
        current_digits+1)[FIBONACCI[current_digits+1]:FIBONACCI[current_digits+2]]
    y_numbers_to_add = np.full(
        (FIBONACCI[current_digits]), 2**(current_digits-1), dtype='int64')

    # print(x_numbers_to_add)
    # print()
    # print(y_numbers_to_add)

    for m in range(1, current_digits-1):
        num_x_partition_digits = current_digits-1-m
        num_y_partition_digits = m

        x_partition = no_consecutive_ones_binary_flipped(
            num_x_partition_digits)
        y_partition = no_consecutive_ones_binary_flipped(
            num_y_partition_digits)
        x_look_up = np.zeros(
            (2**num_x_partition_digits), dtype='int64')
        y_look_up = np.zeros(
            (2**num_y_partition_digits), dtype='int64')

        for n in range(x_partition.size):
            x_look_up[x_partition[n]] = n
        for n in range(y_partition.size):
            y_look_up[y_partition[n]] = n

        counting_matrix = partition_numbers_2d_binary(
            current_digits, num_x_partition_digits, num_y_partition_digits)

        for n in range(x_numbers.size):
            counting_matrix[x_look_up[x_numbers[n] % 2**(num_x_partition_digits)],
                            y_look_up[y_numbers[n] % 2**(num_y_partition_digits)]] -= 1
        # print()
        # print(x_numbers_to_add)
        # print(y_numbers_to_add)
        # print(counting_matrix)

        for n in range(x_numbers_to_add.size):
            # print(
            #     f'{x_look_up[x_numbers_to_add[n] % 2**(num_x_partition_digits)]},{y_look_up[y_numbers_to_add[n] % 2**(num_y_partition_digits)]}')
            if counting_matrix[x_look_up[x_numbers_to_add[n] % 2**(num_x_partition_digits)],
                               y_look_up[y_numbers_to_add[n] % 2**(num_y_partition_digits)]] == 0:
                y_numbers_to_add[n] += 2**(num_y_partition_digits-1)

    final_x_numbers = np.append(x_numbers, x_numbers_to_add, axis=0)
    final_y_numbers = np.append(y_numbers, y_numbers_to_add, axis=0)

    final_x_numbers = np.reshape(
        final_x_numbers, (final_x_numbers.size, 1))
    final_y_numbers = np.reshape(
        final_y_numbers, (final_y_numbers.size, 1))

    # print(final_x_numbers)
    return np.append(final_x_numbers, final_y_numbers, axis=1)


def convert_numbers_to_digits(numbers, num_digits):
    digits = np.zeros((numbers.size, num_digits), dtype='int64')
    for n in range(numbers.size):
        for m in range(num_digits):
            digits[n, m] += (numbers[n] // 2**m) % 2
    return digits


def convert_point_numbers_to_digits(points, num_digits):
    digits = np.zeros((points.shape[0], 2, num_digits), dtype='int64')
    print(digits)
    for n in range(points.shape[0]):
        for m in range(num_digits):
            digits[n, 0, m] += (points[n, 0] // 2**m) % 2
        for m in range(num_digits):
            digits[n, 1, m] += (points[n, 1] // 2**m) % 2
    return digits


##################################################################


def plot_points(points, x_partition, ypartition):
    x = points[:, 0]
    y = points[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=5)
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def plot_points_size_1(points, x_partition, ypartition):
    x = points[:, 0]
    y = points[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=1)
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def plot_points_2(points, num_digits, x_partition, ypartition):
    plt.figure(figsize=(12, 12))  # 9
    ax = plt.gca()
    ax.set_aspect(1)
    # print(fibonacci(num_digits+1))
    plt.plot(points[:fibonacci(num_digits+1), 0],
             points[:fibonacci(num_digits+1), 1], '.', markersize=15, c='k')
    plt.plot(points[fibonacci(num_digits+1):fibonacci(num_digits+2), 0],
             points[fibonacci(num_digits+1):fibonacci(num_digits+2), 1], '.', markersize=15, c='tab:red')
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()
##################################################################


@njit(parallel=True)
def second_part(arr):
    var_sum = 0
    for x in prange(arr.shape[0]):
        for y in prange(arr.shape[0]):
            var_sum += min(1-arr[x, 0], 1-arr[y, 0]) * \
                min(1-arr[x, 1], 1-arr[y, 1])
    return var_sum / arr.shape[0]**2


@njit(parallel=True)
def second_part_version_2(arr):
    value_array = np.array([min(1-arr[x, 0], 1-arr[y, 0]) *
                            min(1-arr[x, 1], 1-arr[y, 1]) for x in range(arr.shape[0]) for y in range(arr.shape[0])])
    return np.sum(value_array) / arr.shape[0]**2


@njit()
def first_part(arr):
    var_sum = 0
    # print()
    for x in range(arr.shape[0]):
        var_sum += (1-arr[x, 0]**2)*(1-arr[x, 1]**2)
    return var_sum / (2 * arr.shape[0])


@njit(parallel=True)
def first_part_version_2(arr):
    value_array = np.array([(1-arr[x, 0])*(1-arr[x, 1])
                           for x in range(arr.shape[0])])
    return np.sum(value_array) / (2 * arr.shape[0])


@njit()
def version_3_first_part(arr):
    var_sum = 0
    # print()
    for x in range(arr.shape[0]):
        var_sum += (1-arr[x, 0]**2)*(1-arr[x, 1]**2)-(2. / 3.)**2
    return var_sum / (2 * arr.shape[0])


@njit(parallel=True)
def version_3_second_part_1(arr):
    var_sum = 0
    for x in prange(arr.shape[0]):
        for y in prange(arr.shape[0]):
            var_sum += min(1-arr[x, 0], 1-arr[y, 0]) * \
                min(1-arr[x, 1], 1-arr[y, 1]) - (1./3.) ** 2
    return var_sum


@njit()
def version_3_second_part_2(arr):
    var_sum = 0
    for x in range(arr.shape[0]):
        var_sum += (1-arr[x, 0])*(1-arr[x, 1]) - (1./2.) ** 2
    return var_sum


@njit()
def version_3_second_part(arr):
    return (version_3_second_part_1(arr)+version_3_second_part_2(arr))/arr.shape[0]**2


def version_3_l2_star_discrepancy(arr):
    np.random.shuffle(arr)
    return (1./arr.shape[0]**2) * ((1./2.)**2 - (1./3.)**2) + \
        -version_3_first_part(arr)+version_3_second_part(arr)


def l2_star_discrepancy(point_set):
    return 1. / 3. ** 2 - first_part(point_set) + second_part(point_set)


def l2_star_discrepancy_version_2(point_set):
    return 1. / 3. ** 2 - first_part_version_2(point_set) + second_part_version_2(point_set)


@ njit(parallel=True)
def version_2_(arr):
    value_array = np.array([min(1-arr[x, 0], 1-arr[y, 0]) *
                            min(1-arr[x, 1], 1-arr[y, 1]) for x in range(arr.shape[0]) for y in range(arr.shape[0])])
    return np.sum(value_array) / arr.shape[0] ** 2


@njit()
def star_discrepancy(point_set):
    point_set = point_set[point_set[:, 0].argsort()]
    n = point_set.shape[0]

    largest_outer_value = 0

    for index_i in range(1, n-1):
        sorted = np.argsort(point_set[:index_i, 1])

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
    sorted = np.argsort(point_set[:index_i, 1])
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


@njit(parallel=True)
def star_discrepancy_parallel(point_set):
    point_set = point_set[point_set[:, 0].argsort()]
    n = point_set.shape[0]

    largest_outer_value = 0

    outer_value_vector = np.zeros((n-1), dtype='float')
    for index_i in prange(1, n-1):
        sorted = np.argsort(point_set[:index_i, 1])

        largest_inner_value = 0

        for index_k in range(index_i-1):
            if largest_inner_value < max(index_k / n - point_set[index_i, 0] * point_set[sorted[index_k], 1],                                    point_set[index_i+1, 0]*point_set[sorted[index_k+1], 1]-index_k / n):
                largest_inner_value = max(index_k / n - point_set[index_i, 0] * point_set[sorted[index_k], 1],
                                          point_set[index_i+1, 0]*point_set[sorted[index_k+1], 1]-index_k / n)
        index_k = index_i
        if largest_inner_value < max(index_i / n - point_set[index_i, 0] * point_set[sorted[-1], 1], point_set[index_i+1, 0]-index_i / n):
            largest_inner_value = max(
                index_i / n - point_set[index_i, 0] * point_set[sorted[-1], 1], point_set[index_i+1, 0]-index_i / n)

        outer_value_vector[index_i] += largest_inner_value
    largest_outer_value = np.max(outer_value_vector)
    index_i = n-1
    sorted = np.argsort(point_set[:index_i, 1])
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

##################################################################


@vectorize()
def random_no_consecutive_ones_number_2(number_of_digits):
    result = 0
    previous_digit = 0
    for n in range(number_of_digits):
        if previous_digit == 0:
            random_binary = random.randint(0, FIBONACCI[20]-1) // FIBONACCI[19]
            result += random_binary*2**n
            previous_digit = random_binary
        else:
            previous_digit = 0
    return result


@vectorize()
def random_no_consecutive_ones_number(number_of_digits):
    result = 0
    rand_0_1 = np.random.uniform(0, 1)

    for m in range(number_of_digits):
        multiplied = rand_0_1 * GOLDEN_RATIO
        digit = int(multiplied)
        result += digit * 2**m
        rand_0_1 = multiplied - digit

    return result


# @njit(parallel=True)
# def random_no_consecutive_ones_point_numbers(arr, number_of_digits):
#     result = np.zeros_like(arr, dtype='float')
#     rand_0_1 = np.random.uniform(0, 1, arr.shape)

#     for m in range(number_of_digits):
#         # print(rand_0_1)
#         multiplied = rand_0_1 * GOLDEN_RATIO
#         # print(multiplied)
#         digit = np.floor(multiplied)
#         # print(digit)
#         # print(digit)
#         result += digit * 2**m
#         # print(result)
#         rand_0_1 = multiplied - digit

#     return result.astype(np.int64)


@njit(parallel=True)
def array_random_no_consecutive_ones_number(length, num_digits):
    point_numbers = np.zeros(2 * length, dtype='int64')
    for n in prange(length*2):
        point_numbers[n] = random_no_consecutive_ones_number(num_digits)


@njit()
def build_x_values(starting_x_numbers, target_num_digits):
    x_numbers = no_consecutive_ones_binary_2(target_num_digits)
    count_array = np.ones(x_numbers[-1]+1, dtype='bool')
    for number in starting_x_numbers:
        count_array[number % 2 ** (target_num_digits)] -= 1
    return x_numbers[count_array[x_numbers]]


@njit()
def partition_numbers_2d_binary_for_random(digits_to_partition, x_partition_digits, y_partition_digits):
    numbers = no_consecutive_ones_binary_2(digits_to_partition)
    x_partition_numbers = numbers[:FIBONACCI[x_partition_digits+2]
                                  ] % 2 ** x_partition_digits
    y_partition_numbers = numbers[:FIBONACCI[y_partition_digits+2]
                                  ] % 2 ** y_partition_digits
    x_partition_last_digit = x_partition_numbers % 2
    y_partition_last_digit = y_partition_numbers % 2

    # print(x_partition_numbers[-1]+1)
    counting_matrix = np.zeros(
        (x_partition_numbers[-1]+1, y_partition_numbers[-1]+1), dtype='int64')
    for m in range(x_partition_numbers.shape[0]):
        for n in range(y_partition_numbers.shape[0]):
            counting_matrix[x_partition_numbers[m], y_partition_numbers[n]] = FIBONACCI[digits_to_partition + 2 - x_partition_digits -
                                                                                        y_partition_digits - x_partition_last_digit[m]-y_partition_last_digit[n]]
    return counting_matrix

    # last_digit_x = np.zeros((FIBONACCI[x_partition_digits+2]), dtype='int')
    # last_digit_x[1] = 1
    # for m in range(2, x_partition_digits+1):
    #     last_digit_x[FIBONACCI[m+1]:FIBONACCI[m+2]] += np.add(
    #         last_digit_x[FIBONACCI[m+1]:FIBONACCI[m+2]], last_digit_x[:FIBONACCI[m]])

    # last_digit_y = np.zeros((FIBONACCI[y_partition_digits+2]), dtype='int')
    # last_digit_y[1] = 1
    # for m in range(2, y_partition_digits+1):
    #     last_digit_y[FIBONACCI[m+1]:FIBONACCI[m+2]] += np.add(
    #         last_digit_y[FIBONACCI[m+1]:FIBONACCI[m+2]], last_digit_y[:FIBONACCI[m]])

    # return np.array([[FIBONACCI[digits_to_partition+2-x_partition_digits-y_partition_digits-r-c] for r in last_digit_y] for c in last_digit_x], dtype='int')


@njit()
def randomize_tail_end(number, relavent_digits):

    last_digit = number // 2 ** (relavent_digits-1)  # arrays indexed from 0
    # arrays indexed from 0, so -1 is not needed
    first_possible_one = relavent_digits+last_digit
    # number += random.randint(0, 1) * 2 ** first_possible_one
    # relavent_digits += 1
    last_digit = number // 2 ** (relavent_digits-1)  # arrays indexed from 0
    # arrays indexed from 0, so -1 is not needed
    first_possible_one = relavent_digits+last_digit
    tail_end = random_no_consecutive_ones_number(32-first_possible_one)
    return number + tail_end * 2 ** first_possible_one


@njit()
def extend_points_randomized(point_numbers, starting_digits):
    x_numbers = point_numbers[:, 0]
    y_numbers = point_numbers[:, 1]
    current_digits = starting_digits+1
    x_numbers_to_add = build_x_values(x_numbers, current_digits)
    y_numbers_to_add = np.zeros(
        (FIBONACCI[current_digits]), dtype='int64')

    # print(x_numbers_to_add)
    # # print()
    # print(y_numbers_to_add)

    for m in range(1, current_digits):
        num_x_partition_digits = current_digits-1-m
        num_y_partition_digits = m

        x_partition = no_consecutive_ones_binary_flipped(
            num_x_partition_digits)
        y_partition = no_consecutive_ones_binary_flipped(
            num_y_partition_digits)
        x_look_up = np.zeros(
            (2**num_x_partition_digits), dtype='int64')
        y_look_up = np.zeros(
            (2**num_y_partition_digits), dtype='int64')

        for n in range(x_partition.size):
            x_look_up[x_partition[n]] = n
        for n in range(y_partition.size):
            y_look_up[y_partition[n]] = n

        counting_matrix = partition_numbers_2d_binary(
            current_digits, num_x_partition_digits, num_y_partition_digits)
        # print(counting_matrix)

        for n in range(x_numbers.size):
            counting_matrix[x_look_up[x_numbers[n] % 2**(num_x_partition_digits)],
                            y_look_up[y_numbers[n] % 2**(num_y_partition_digits)]] -= 1

        # print(x_numbers_to_add)
        # print(y_numbers_to_add)
        # print(counting_matrix)
        # print()
        for n in range(x_numbers_to_add.size):
            # print(
            #     f'{x_look_up[x_numbers_to_add[n] % 2**(num_x_partition_digits)]},{y_look_up[y_numbers_to_add[n] % 2**(num_y_partition_digits)]}')
            if counting_matrix[x_look_up[x_numbers_to_add[n] % 2**(num_x_partition_digits)],
                               y_look_up[y_numbers_to_add[n] % 2**(num_y_partition_digits)]] == 0:
                y_numbers_to_add[n] += 2**(num_y_partition_digits-1)

    num_x_partition_digits = 0
    num_y_partition_digits = current_digits

    x_partition = no_consecutive_ones_binary_flipped(
        num_x_partition_digits)
    y_partition = no_consecutive_ones_binary_flipped(
        num_y_partition_digits)
    x_look_up = np.zeros(
        (2**num_x_partition_digits), dtype='int64')
    y_look_up = np.zeros(
        (2**num_y_partition_digits), dtype='int64')

    for n in range(x_partition.size):
        x_look_up[x_partition[n]] = n
    for n in range(y_partition.size):
        y_look_up[y_partition[n]] = n

    counting_matrix = partition_numbers_2d_binary(
        current_digits, num_x_partition_digits, num_y_partition_digits)
    # print(counting_matrix)

    for n in range(x_numbers.size):
        counting_matrix[x_look_up[x_numbers[n] % 2**(num_x_partition_digits)],
                        y_look_up[y_numbers[n] % 2**(num_y_partition_digits)]] -= 1

    # print(x_numbers_to_add)
    # print(y_numbers_to_add)
    # print(counting_matrix)
    # print()
    for n in range(x_numbers_to_add.size):
        # print(
        #     f'{x_look_up[x_numbers_to_add[n] % 2**(num_x_partition_digits)]},{y_look_up[y_numbers_to_add[n] % 2**(num_y_partition_digits)]}')
        if counting_matrix[x_look_up[x_numbers_to_add[n] % 2**(num_x_partition_digits)],
                           y_look_up[y_numbers_to_add[n] % 2**(num_y_partition_digits)]] == 0:
            y_numbers_to_add[n] += 2**(num_y_partition_digits-1)

    for n in range(x_numbers_to_add.shape[0]):
        x_numbers_to_add[n] = randomize_tail_end(
            x_numbers_to_add[n], current_digits)

    for n in range(y_numbers_to_add.shape[0]):
        y_numbers_to_add[n] = randomize_tail_end(
            y_numbers_to_add[n], current_digits)

    # for n in range(y_numbers_to_add.shape[0]):
    #     last_digit = (y_numbers_to_add[n] // 2**(current_digits-1)) % 2
    #     y_numbers_to_add = np.add(y_numbers_to_add, np.multiply(random_no_consecutive_ones_number(
    #         32 - current_digits - last_digit - 1), 2 ** (current_digits + last_digit)))

    # print(x_numbers.shape)
    # print(x_numbers_to_add.shape)
    final_x_numbers = np.append(x_numbers, x_numbers_to_add, axis=0)
    final_y_numbers = np.append(y_numbers, y_numbers_to_add, axis=0)

    final_x_numbers = np.reshape(
        final_x_numbers, (final_x_numbers.size, 1))
    final_y_numbers = np.reshape(
        final_y_numbers, (final_y_numbers.size, 1))

    # print(final_x_numbers)
    return np.append(final_x_numbers, final_y_numbers, axis=1)


@njit()
def generate_random_sequence(num_digits):

    first_point = np.array(
        [[random.randint(0, 1), random.randint(0, 1)]], dtype='int64')
    # print(first_point)
    first_point[0, 0] = randomize_tail_end(first_point[0, 0], 1)
    first_point[0, 1] = randomize_tail_end(first_point[0, 1], 1)
    # first_point = np.array([[random_no_consecutive_ones_number(
    #     32), random_no_consecutive_ones_number(32)]], dtype='int64')
    # print(first_point)

    if num_digits == 0:
        return first_point

    second_point = np.array(
        [[1-(first_point[0, 0] % 2), 1-(first_point[0, 1] % 2)]], dtype='int64')
    if second_point[0, 0] == 0:
        second_point[0, 0] += random_no_consecutive_ones_number(
            31) * 2
    else:
        second_point[0, 0] += random_no_consecutive_ones_number(
            30) * 4
    if second_point[0, 1] == 0:
        second_point[0, 1] += random_no_consecutive_ones_number(
            31) * 2
    else:
        second_point[0, 1] += random_no_consecutive_ones_number(
            30) * 4
    point_numbers_2 = np.append(first_point, second_point, axis=0)
    # plot_points(points_1, 1, 1)
    # plot_points(points_2, 2, 2)

    # num_digits = 5
    # x_numbers = no_consecutive_ones_binary_flipped(num_digits)
    # y_numbers = no_consecutive_ones_binary_2(num_digits)
    # point_numbers = np.array([[x_numbers[n], y_numbers[n]]
    #                          for n in range(x_numbers.shape[0])], dtype='int64')
    # print(point_numbers)

    for m in range(1, num_digits):
        point_numbers_2 = extend_points_randomized(point_numbers_2, m)

    return point_numbers_2


@njit()
def generate_n_scrambled_sequences(n, num_digits):
    # k = num_digits
    # point_numbers_1 = generate_random_sequence(k)
    point_numbers = np.zeros((FIBONACCI[num_digits+2]*n, 2), dtype='int64')
    # print(point_numbers.shape)
    # print(FIBONACCI[num_digits+2])
    # print(generate_random_sequence(num_digits).shape)

    for k in prange(n):
        # temp_points = generate_random_sequence(k)
        # point_numbers_1 = np.append(point_numbers_1, temp_points, axis=0)
        # print(
        #     f'{k} : {FIBONACCI[num_digits+2]*n}: {FIBONACCI[num_digits+2] * (n+1)}')
        point_numbers[FIBONACCI[num_digits+2]*k:FIBONACCI[num_digits+2]
                      * (k+1)] += generate_random_sequence(num_digits)
    return point_numbers


# @njit()
def digits_in_common_log(int_1, int_2):
    arr = np.sort(np.array([integer_to_decimal(int_1),
                            integer_to_decimal(int_2)]))
    # print(arr)
    multiplied = arr[1]-arr[0] + GOLDEN_RATIO ** (-34)
    # print(multiplied)
    result = math.log2(1 / multiplied) / LOG_GOLDEN_RATIO
    # for j in range(32):
    #     multiplied = multiplied * GOLDEN_RATIO
    #     if multiplied + GOLDEN_RATIO ** (-34) >= 1:
    #         break
    # if j == 31:
    #     return 0
    # else:
    #     return j
    result = int(math.ceil(result))
    if result == 34:
        return 0
    else:
        return result

# @njit()


# @njit(parallel=True)
def make_table_digits_in_common_numpy(num_digits):
    numbers = kronecker_numbers_sorted(num_digits)
    decimals = np.array([integer_to_decimal(n)
                        for n in numbers], dtype='f4')
    results_table = np.zeros(
        (FIBONACCI[num_digits+2], FIBONACCI[num_digits+2]), dtype='int8')
    adder = GOLDEN_RATIO ** (-34)
    for n in prange(FIBONACCI[num_digits+2]):
        for k in range(n+1, FIBONACCI[num_digits+2]):
            decimal = decimals[n] - decimals[k]
            # decimal_table = np.array([decimals - x for x in decimals], dtype='f4')
            # decimal = abs(decimal)
            # decimal += adder
            decimal = - math.log(abs(decimal)+adder) / LOG_GOLDEN_RATIO_2
            results_table[n, k] = math.ceil(decimal)
    for n in prange(1, FIBONACCI[num_digits+2]):
        for k in range(n):
            results_table[n, k] = results_table[k, n]
    return results_table


@njit
def make_look_up_table(num_digits):
    look_up = np.zeros(
        (2**num_digits), dtype='int64')
    numbers = no_consecutive_ones_binary_2(num_digits)
    for n in range(FIBONACCI[num_digits+2]):
        look_up[numbers[n]] = n
    return look_up


def find_fibonacci_index(num):
    for n in range(FIBONACCI.shape[0]):
        if FIBONACCI[n] == num:
            break
    return n


path = Path('~\Documents\PythonProjects').expanduser()
path.mkdir(parents=True, exist_ok=True)
# np.save(path/'digits_in_common_23', table_1_3)
DIGITS_IN_COMMON_TABLE = np.load(path/'digits_in_common_23.npy')


# @jit()
def scramble_numbers(numbers):

    possibilities = no_consecutive_ones_binary_2(23)
    look_up = make_look_up_table(23)
    # num_digits = find_fibonacci_index(numbers.shape[0])

    for m in range(2000):
        scrambled_numbers = np.zeros(numbers.shape[0], dtype='int64')
        scrambled_numbers[0] = np.copy(
            possibilities[random.randint(0, possibilities.shape[0]+1)])
        truth = True
        for n in range(numbers.shape[0]):
            remaining_possibilities_numbers = np.copy(possibilities)
            remaining_possibilities_digits_in_common = np.copy(
                DIGITS_IN_COMMON_TABLE[look_up[numbers[n]]])
            for k in range(n):
                print(f'm = {m} : n = {n} : k = {k}')
                digits_in_common = DIGITS_IN_COMMON_TABLE[look_up[numbers[n]],
                                                          look_up[numbers[k]]]
                valid_possibilities = np.zeros_like(
                    remaining_possibilities_digits_in_common, dtype='bool')
                for j in range(valid_possibilities.shape[0]):
                    if remaining_possibilities_digits_in_common[j] == digits_in_common:
                        valid_possibilities[j] = 1
                remaining_possibilities_numbers = remaining_possibilities_numbers[
                    valid_possibilities]
                remaining_possibilities_digits_in_common = remaining_possibilities_digits_in_common[
                    valid_possibilities]
                if remaining_possibilities_numbers.shape[0] == 0:
                    truth = False
                    break
            if not truth:
                break
            # print(f'{n} : {remaining_possibilities_numbers.shape[0]}')
            scrambled_numbers[n] = np.copy(remaining_possibilities_numbers[random.randint(
                0, remaining_possibilities_numbers.shape[0]+1)])
        if truth:
            return scrambled_numbers
        # print()

    return numbers

# @jit()


@njit(parallel=True)
def scramble_numbers_2(numbers):

    possibilities = no_consecutive_ones_binary_2(23)
    look_up = make_look_up_table(23)
    # num_digits = find_fibonacci_index(numbers.shape[0])

    for m in range(1000):
        print(m)
        scrambled_numbers = np.zeros(numbers.shape[0], dtype='int64')
        scrambled_numbers[0] = possibilities[random.randint(
            0, possibilities.shape[0]+1)]
        truth = True
        for n in range(numbers.shape[0]):
            valid_possibilities = np.ones_like(possibilities, dtype='bool')
            for k in range(n):
                # print(f'm = {m} : n = {n} : k = {k}')
                digits_in_common = DIGITS_IN_COMMON_TABLE[look_up[numbers[n]],
                                                          look_up[numbers[k]]]-1
                digits_in_common_row = DIGITS_IN_COMMON_TABLE[look_up[scrambled_numbers[k]]]-1
                for j in prange(valid_possibilities.shape[0]):
                    if digits_in_common_row[j] == digits_in_common:
                        valid_possibilities[j] = 0

            valid_numbers = possibilities[valid_possibilities]
            if valid_numbers.shape[0] == 0:
                truth = False
                break
            scrambled_numbers[n] = valid_numbers[random.randint(
                0, valid_numbers.shape[0]+1)]
        if truth:
            return scrambled_numbers
        # print()

    return np.copy(numbers)


@njit()
def randomize_many(m):
    num_digits = 4
    x_numbers = no_consecutive_ones_binary_2(num_digits)
    sort_order = np.argsort(convert_integer_points_to_decimals(x_numbers))
    y_numbers = no_consecutive_ones_binary_flipped(num_digits)

    x_numbers_1 = no_consecutive_ones_binary_2(num_digits)
    y_numbers_1 = no_consecutive_ones_binary_flipped(num_digits)

    for k in range(m):
        y_numbers_temp = scramble_numbers_2(y_numbers_1)
        x_numbers_temp = scramble_numbers_2(x_numbers_1)

        x_numbers = np.append(x_numbers, x_numbers_temp)
        y_numbers = np.append(y_numbers, y_numbers_temp)
        # print(x_numbers)
        print(k)
    point_numbers = np.array([[x_numbers[n], y_numbers[n]]
                              for n in range(x_numbers.shape[0])], dtype='int64')
    return point_numbers


def kronecker_numbers(m):
    decimals = np.array([n*GOLDEN_RATIO**-1 - math.floor(n*GOLDEN_RATIO**-1) +
                        GOLDEN_RATIO**-32 for n in range(FIBONACCI[m+2])], dtype='float')
    numbers = np.zeros(FIBONACCI[m+2], dtype='float')
    # decimals = decimals +  GOLDEN_RATIO**-32
    print(decimals)
    for m in range(30):
        decimals = decimals * GOLDEN_RATIO
        print(f'{m}:{decimals}')
        integer_part = np.floor(decimals)
        print(f'{m}:{integer_part}')
        numbers += integer_part*2**m
        decimals = decimals - integer_part
        print(decimals)
        print()
    return np.int64(numbers)


def kronecker_numbers_sorted(m):
    decimals = np.sort(np.array([n*GOLDEN_RATIO**-1 - math.floor(n*GOLDEN_RATIO**-1) +
                       GOLDEN_RATIO**-32 for n in range(FIBONACCI[m+2])], dtype='float'))
    numbers = np.zeros(FIBONACCI[m+2], dtype='float')
    # decimals = decimals +  GOLDEN_RATIO**-32
    # print(decimals)
    for m in range(30):
        decimals = decimals * GOLDEN_RATIO
        # print(f'{m}:{decimals}')
        integer_part = np.floor(decimals)
        # print(f'{m}:{integer_part}')
        numbers += integer_part*2**m
        decimals = decimals - integer_part
        # print(decimals)
        # print()
    return np.int64(numbers)


def convert_numbers_to_digits(arr):
    for n in range(32):
        if np.sum(arr // 2**n) == 0:
            break
    digits = np.zeros((arr.shape[0], n), dtype='int8')
    for k in range(n):
        digits[:, k] = (arr // 2**k) % 2
    return digits


print(convert_numbers_to_digits(kronecker_numbers(4)))

# print(make_table_digits_in_common_numpy(3)-1)
