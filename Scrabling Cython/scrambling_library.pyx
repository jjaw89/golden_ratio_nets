import cupy as cp
import random
from cmath import sqrt
from msilib.schema import Binary
from tkinter import S
from xml.dom import WrongDocumentErr
import numpy as np
import time
import sys
import math
import scipy.special
from scipy.stats import qmc
from numba import jit, njit, vectorize, prange
import concurrent.futures
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sympy import GoldenRatio, plot
plt.style.use('seaborn-whitegrid')

GOLDEN_RATIO = ((1+np.sqrt(5))/2).real

# # The seeds for s10, s11, s12 must be integers in [0, m1 - 1] and not all 0.
# # The seeds for s20, s21, s22 must be integers in [0, m2 - 1] and not all 0.

# # SEED = 12345
# s_10 = 12345
# s_11 = 12345
# s_12 = 12345
# s_20 = 12345
# s_21 = 12345
# s_22 = 12345


# def random_between_0_1():
#     global s_10
#     global s_11
#     global s_12
#     global s_20
#     global s_21
#     global s_22

#     norm = 2.328306549295728e-10
#     m_1 = 4294967087.0
#     m_2 = 4294944443.0
#     a_12 = 1403580.0
#     a_13_n = 810728.0
#     a_21 = 527612.0
#     a_23_n = 1370589.0

#     # Component 1
#     # print(s_10)
#     p_1 = a_12 * s_11 - a_13_n * s_10
#     k = p_1 // m_1
#     p_1 -= k * m_1
#     # print(m_1)
#     if p_1 < 0.0:
#         p_1 += m_1
#     s_10 = s_11
#     s_11 = s_12
#     s_12 = p_1

#     # Component 2
#     p_2 = a_21 * s_22 - a_23_n * s_20
#     k = p_2 // m_2
#     p_2 -= k * m_2
#     if p_2 < 0.0:
#         p_2 += m_2
#     s_20 = s_21
#     s_21 = s_22
#     s_22 = p_2

#     # Combination
#     if p_1 <= p_2:
#         return ((p_1 - p_2 + m_1) * norm)
#     else:
#         return ((p_1 - p_2) * norm)


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
        numbers[fibonacci(m+1):fibonacci(m+2)] += np.add(numbers[fibonacci(m+1)                                                                 :fibonacci(m+2)], numbers[:fibonacci(m)])+2**(m-1)
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
    return np.array([integer_to_decimal(n) for n in numbers])


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


@njit()
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
    plt.plot(x, y, '.', markersize=1)
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def plot_points_point_size_5(points, x_partition, ypartition):
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


def plot_points_point_size_10(points, x_partition, ypartition):
    x = points[:, 0]
    y = points[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=10)
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

#######################################


def random_0_or_1(prob):
    if random.uniform(0, 1) <= prob:
        return 0
    else:
        return 1


# start_time = time.perf_counter()

# end_time = time.perf_counter()
# print(f'Total_time : {end_time-start_time:0.6f}')

# @njit(parallel=True)
def scramble_numbers_m_digits_2(m, numbers):

    m = m-1

    start_time = time.perf_counter()
    # numbers_to_split = no_consecutive_ones_binary_2(m)
    original_order = no_consecutive_ones_binary_2(m+1)
    end_time = time.perf_counter()
    print(f'generate 2  time : {end_time-start_time:0.6f}')

    start_time = time.perf_counter()
    random_zeros_and_ones = np.random.randint(
        2, size=FIBONACCI[m+1], dtype='int64')
    end_time = time.perf_counter()
    print(f'random time : {end_time-start_time:0.6f}')

    start_time = time.perf_counter()
    permutation = np.zeros(FIBONACCI[m+3], dtype='int64')
    permutation[:FIBONACCI[m+2]] += original_order[:FIBONACCI[m+2]]
    permutation[FIBONACCI[m+2]:] += original_order[:FIBONACCI[m+1]
                                                   ] + (1-random_zeros_and_ones) * 2**m
    permutation[:FIBONACCI[m+1]] += random_zeros_and_ones * 2**m
    end_time = time.perf_counter()
    print(f'permutation time : {end_time-start_time:0.6f}')

    start_time = time.perf_counter()
    look_up_array = np.zeros(original_order[-1]+1, dtype='int64')
    look_up_array[original_order] = permutation
    end_time = time.perf_counter()
    print(f'look up time : {end_time-start_time:0.6f}')

    start_time = time.perf_counter()
    digits_to_permute = numbers % 2**(m+1)
    permuted_digits = np.array([look_up_array[n]
                               for n in digits_to_permute], dtype='int64')
    end_time = time.perf_counter()
    print(f'permuted digits time : {end_time-start_time:0.6f}')

    start_time = time.perf_counter()
    mask = (numbers // 2**(m+1)) % 2 == 1
    permuted_digits[mask] = digits_to_permute[mask]
    numbers = (numbers // 2**(m+1))*2**(m+1) + permuted_digits
    end_time = time.perf_counter()
    print(f'final time : {end_time-start_time:0.6f}')

    return numbers


def scramble_numbers_m_digits(m, numbers):
    m = m-1

    original_order = no_consecutive_ones_binary_2(m+1)

    random_zeros_and_ones = np.random.randint(
        2, size=FIBONACCI[m+1], dtype='int64')

    permutation = np.zeros(FIBONACCI[m+3], dtype='int64')
    permutation[:FIBONACCI[m+2]] += original_order[:FIBONACCI[m+2]]
    permutation[FIBONACCI[m+2]:] += original_order[:FIBONACCI[m+1]
                                                   ] + (1-random_zeros_and_ones) * 2**m
    permutation[:FIBONACCI[m+1]] += random_zeros_and_ones * 2**m

    look_up_array = np.zeros(original_order[-1]+1, dtype='int64')
    look_up_array[original_order] = permutation

    digits_to_permute = numbers % 2**(m+1)
    permuted_digits = np.array([look_up_array[n]
                               for n in digits_to_permute], dtype='int64')

    mask = (numbers // 2**(m+1)) % 2 == 1
    permuted_digits[mask] = digits_to_permute[mask]
    numbers = (numbers // 2**(m+1))*2**(m+1) + permuted_digits

    return numbers


def scramble_numbers_m_digits_unequal_chance(m, numbers):
    m = m-1

    original_order = no_consecutive_ones_binary_2(m+1)

    random_zeros_and_ones = np.random.randint(
        FIBONACCI[15], size=FIBONACCI[m+1], dtype='int64') // FIBONACCI[14]

    permutation = np.zeros(FIBONACCI[m+3], dtype='int64')
    permutation[:FIBONACCI[m+2]] += original_order[:FIBONACCI[m+2]]
    permutation[FIBONACCI[m+2]:] += original_order[:FIBONACCI[m+1]
                                                   ] + (1-random_zeros_and_ones) * 2**m
    permutation[:FIBONACCI[m+1]] += random_zeros_and_ones * 2**m

    look_up_array = np.zeros(original_order[-1]+1, dtype='int64')
    look_up_array[original_order] = permutation

    digits_to_permute = numbers % 2**(m+1)
    permuted_digits = np.array([look_up_array[n]
                               for n in digits_to_permute], dtype='int64')

    mask = (numbers // 2**(m+1)) % 2 == 1
    permuted_digits[mask] = digits_to_permute[mask]
    numbers = (numbers // 2**(m+1))*2**(m+1) + permuted_digits

    return numbers


def scramble_numbers_m_digits_unequal_chance_sqrt(m, numbers):
    m = m-1

    original_order = no_consecutive_ones_binary_2(m+1)

    random_zeros_and_ones = np.random.randint(
        8119, size=FIBONACCI[m+1], dtype='int64') // 5741
    random_zeros_and_ones = random_zeros_and_ones // 1
    # random_zeros_and_ones = np.random.randint(
    #     13860, size=FIBONACCI[m+1], dtype='int64') // 13860 - 8119

    # random_zeros_and_ones = np.random.randint(
    #     2, size=FIBONACCI[m+1], dtype='int64')

    permutation = np.zeros(FIBONACCI[m+3], dtype='int64')
    permutation[:FIBONACCI[m+2]] += original_order[:FIBONACCI[m+2]]
    permutation[FIBONACCI[m+2]:] += original_order[:FIBONACCI[m+1]
                                                   ] + (1-random_zeros_and_ones) * 2**m
    permutation[:FIBONACCI[m+1]] += random_zeros_and_ones * 2**m

    look_up_array = np.zeros(original_order[-1]+1, dtype='int64')
    look_up_array[original_order] = permutation

    digits_to_permute = numbers % 2**(m+1)
    permuted_digits = np.array([look_up_array[n]
                               for n in digits_to_permute], dtype='int64')

    mask = (numbers // 2**(m+1)) % 2 == 1
    permuted_digits[mask] = digits_to_permute[mask]
    numbers = (numbers // 2**(m+1))*2**(m+1) + permuted_digits

    return numbers


@njit()
def rand_vect(length):
    vect = [random.randint(0, 1)]
    for _ in range(length-1):
        vect += [random.randint(0, 1)]
    return np.array(vect, dtype='int64')


def scramble_numbers(numbers, up_to):
    for m in range(1, up_to):
        numbers[:, 0] = scramble_numbers_m_digits_unequal_chance(
            m, numbers[:, 0])
    for m in range(1, up_to):
        numbers[:, 1] = scramble_numbers_m_digits_unequal_chance(
            m, numbers[:, 1])
    return numbers


def scramble_numbers_2_unequal_chance(numbers, up_to):
    numbers[:, 0] = scramble_numbers_m_digits_unequal_chance(1, numbers[:, 0])
    for m in range(2, up_to):
        numbers[:, 0] = scramble_numbers_m_digits_unequal_chance(
            m, numbers[:, 0])
        numbers[:, 0] = scramble_numbers_m_digits_unequal_chance(
            m-1, numbers[:, 0])

    numbers[:, 1] = scramble_numbers_m_digits_unequal_chance(1, numbers[:, 1])
    for m in range(2, up_to):
        numbers[:, 1] = scramble_numbers_m_digits_unequal_chance(
            m, numbers[:, 1])
        numbers[:, 1] = scramble_numbers_m_digits_unequal_chance(
            m-1, numbers[:, 1])
    return numbers


def scramble_numbers_2_equal_chance(numbers, up_to):
    numbers[:, 0] = scramble_numbers_m_digits(1, numbers[:, 0])
    for m in range(2, up_to):
        numbers[:, 0] = scramble_numbers_m_digits(
            m, numbers[:, 0])
        numbers[:, 0] = scramble_numbers_m_digits(
            m-1, numbers[:, 0])

    numbers[:, 1] = scramble_numbers_m_digits(1, numbers[:, 1])
    for m in range(2, up_to):
        numbers[:, 1] = scramble_numbers_m_digits(
            m, numbers[:, 1])
        numbers[:, 1] = scramble_numbers_m_digits(
            m-1, numbers[:, 1])
    return numbers


def scramble_numbers_first_backtracking_then_not(numbers):
    for n in range(32):
        if np.sum(numbers // 2 ** n) == 0:
            break
    # print(n)
    numbers[:, 0] = scramble_numbers_m_digits(1, numbers[:, 0])
    for m in range(2, n+1):
        numbers[:, 0] = scramble_numbers_m_digits(
            m, numbers[:, 0])
        numbers[:, 0] = scramble_numbers_m_digits(
            m-1, numbers[:, 0])
    numbers[:, 0] = scramble_numbers_m_digits(n+1, numbers[:, 0])
    for m in range(n+2, 31):
        numbers[:, 0] = scramble_numbers_m_digits_unequal_chance(
            m, numbers[:, 0])

    numbers[:, 1] = scramble_numbers_m_digits(1, numbers[:, 1])
    for m in range(2, n+1):
        numbers[:, 1] = scramble_numbers_m_digits(
            m, numbers[:, 1])
        numbers[:, 1] = scramble_numbers_m_digits(
            m-1, numbers[:, 1])
    numbers[:, 1] = scramble_numbers_m_digits(n+1, numbers[:, 1])
    for m in range(n+2, 31):
        numbers[:, 1] = scramble_numbers_m_digits_unequal_chance(
            m, numbers[:, 1])

    return numbers


def scramble_numbers_first_backtracking_then_not_2(numbers):
    for n in range(32):
        if np.sum(numbers // 2 ** n) == 0:
            break
    # print(n)
    if n == 1:
        numbers[:, 0] = scramble_numbers_m_digits(1, numbers[:, 0])
    else:
        numbers[:, 0] = scramble_numbers_m_digits_unequal_chance_sqrt(
            1, numbers[:, 0])
        for m in range(2, n+1):
            numbers[:, 0] = scramble_numbers_m_digits_unequal_chance_sqrt(
                m, numbers[:, 0])
            numbers[:, 0] = scramble_numbers_m_digits_unequal_chance_sqrt(
                m-1, numbers[:, 0])
        numbers[:, 0] = scramble_numbers_m_digits_unequal_chance_sqrt(
            n+1, numbers[:, 0])

    for m in range(n+2, 31):
        numbers[:, 0] = scramble_numbers_m_digits_unequal_chance(
            m, numbers[:, 0])

    if n == 1:
        numbers[:, 1] = scramble_numbers_m_digits(1, numbers[:, 1])
    else:
        numbers[:, 1] = scramble_numbers_m_digits_unequal_chance_sqrt(
            1, numbers[:, 1])
        for m in range(2, n+1):
            numbers[:, 1] = scramble_numbers_m_digits_unequal_chance_sqrt(
                m, numbers[:, 1])
            numbers[:, 1] = scramble_numbers_m_digits_unequal_chance_sqrt(
                m-1, numbers[:, 1])
        numbers[:, 1] = scramble_numbers_m_digits_unequal_chance_sqrt(
            n+1, numbers[:, 1])

    for m in range(n+2, 31):
        numbers[:, 1] = scramble_numbers_m_digits_unequal_chance(
            m, numbers[:, 1])

    return numbers


# num_digits = 8

# y_numbers = no_consecutive_ones_binary_flipped(num_digits)
# x_numbers = no_consecutive_ones_binary_2(num_digits)
# point_numbers = np.array([[x_numbers[n], y_numbers[n]]
#                          for n in range(x_numbers.shape[0])], dtype='int64')

# # # scramble_numbers_first_backtracking_then_not(point_numbers)

# # point_numbers = np.array([[0, 0], [1, 1], [2, 2]], dtype='int64')

# scrambled_points = scramble_numbers_first_backtracking_then_not_2(
#     np.copy(point_numbers))
# scrambled = np.copy(scrambled_points)
# scrambled_points = scramble_numbers_first_backtracking_then_not_2(
#     scrambled_points)
# scrambled_2 = np.copy(scrambled_points)

# # print(scrambled)
# # scrambled_0 = np.array([scrambled[0]])
# # # print(scrambled_0)
# # scrambled_1 = np.array([scrambled[1]])
# # scrambled_2 = np.array([scrambled[2]])
# start_time = time.perf_counter()
# for m in range(500):
#     scrambled_points = scramble_numbers_first_backtracking_then_not_2(
#         np.copy(point_numbers))
#     scrambled = np.append(scrambled, np.copy(scrambled_points), axis=0)
#     scrambled_points = scramble_numbers_first_backtracking_then_not_2(
#         scrambled_points)
#     scrambled_2 = np.append(scrambled_2, scrambled_points, axis=0)
#     # scrambled_0 = np.append(scrambled_0, np.array([scrambled[0]]), axis=0)
#     # scrambled_1 = np.append(scrambled_1, np.array([scrambled[1]]), axis=0)
#     # scrambled_2 = np.append(scrambled_2, np.array([scrambled[2]]), axis=0)

#     if m % 25 == 0:
#         mid_time = time.perf_counter() - start_time
#         print(f'{m} : time so far {mid_time}')
# # print(scrambled_0)
# # points_0 = convert_integer_points_to_decimals(scrambled_0)
# # points_1 = convert_integer_points_to_decimals(scrambled_1)
# # points_2 = convert_integer_points_to_decimals(scrambled_2)
# # plot_points_point_size_10(points_0, 0, 0)
# # plot_points_point_size_10(points_1, 0, 0)
# # plot_points_point_size_10(points_2, 0, 0)

# points = convert_integer_points_to_decimals(scrambled)
# plot_points(points, 0, 0)
# plot_points_point_size_5(points, 0, 0)
# plot_points_point_size_10(points, 0, 0)

# points = convert_integer_points_to_decimals(scrambled_2)
# plot_points(points, 0, 0)
# plot_points_point_size_5(points, 0, 0)
# plot_points_point_size_10(points, 0, 0)
########################################################

num_digits = 8

y_numbers = no_consecutive_ones_binary_flipped(num_digits)
x_numbers = no_consecutive_ones_binary_2(num_digits)
point_numbers = np.array([[x_numbers[n], y_numbers[n]]
                         for n in range(x_numbers.shape[0])], dtype='int64')

# # scramble_numbers_first_backtracking_then_not(point_numbers)

# point_numbers = np.array([[0, 0], [1, 1], [2, 2]], dtype='int64')

scrambled_points = scramble_numbers(
    np.copy(point_numbers), 30)
scrambled = np.copy(scrambled_points)
scrambled_points = scramble_numbers(
    scrambled_points, 30)
scrambled_2 = np.copy(scrambled_points)

# print(scrambled)
# scrambled_0 = np.array([scrambled[0]])
# # print(scrambled_0)
# scrambled_1 = np.array([scrambled[1]])
# scrambled_2 = np.array([scrambled[2]])
start_time = time.perf_counter()
for m in range(1):
    scrambled_points = scramble_numbers(
        np.copy(point_numbers), 30)
    scrambled = np.append(scrambled, np.copy(scrambled_points), axis=0)
    scrambled_points = scramble_numbers(
        scrambled_points, 30)
    scrambled_2 = np.append(scrambled_2, scrambled_points, axis=0)
    # scrambled_0 = np.append(scrambled_0, np.array([scrambled[0]]), axis=0)
    # scrambled_1 = np.append(scrambled_1, np.array([scrambled[1]]), axis=0)
    # scrambled_2 = np.append(scrambled_2, np.array([scrambled[2]]), axis=0)

    if m % 25 == 0:
        mid_time = time.perf_counter() - start_time
        print(f'{m} : time so far {mid_time}')
# print(scrambled_0)
# points_0 = convert_integer_points_to_decimals(scrambled_0)
# points_1 = convert_integer_points_to_decimals(scrambled_1)
# points_2 = convert_integer_points_to_decimals(scrambled_2)
# plot_points_point_size_10(points_0, 0, 0)
# plot_points_point_size_10(points_1, 0, 0)
# plot_points_point_size_10(points_2, 0, 0)

points = convert_integer_points_to_decimals(scrambled)
plot_points(points, 0, 0)
plot_points_point_size_5(points, 0, 0)
plot_points_point_size_10(points, 0, 0)

points = convert_integer_points_to_decimals(scrambled_2)
plot_points(points, 0, 0)
plot_points_point_size_5(points, 0, 0)
plot_points_point_size_10(points, 0, 0)
########################################################

# vector_results_1 = []
# vector_results_2 = []
# for num_digits in range(1, 19):  # 22
#     start_time = time.perf_counter()

#     result_1 = [FIBONACCI[num_digits+2]]
#     result_2 = [FIBONACCI[num_digits+2]]
#     for _ in range(10):
#         y_numbers = no_consecutive_ones_binary_flipped(num_digits)
#         x_numbers = no_consecutive_ones_binary_2(num_digits)
#         point_numbers = np.array([[x_numbers[n], y_numbers[n]]
#                                   for n in range(x_numbers.shape[0])])

#         point_numbers = scramble_numbers_first_backtracking_then_not(
#             point_numbers)

#         point_numbers = scramble_numbers_first_backtracking_then_not(
#             point_numbers)

#         point_set = convert_integer_points_to_decimals(point_numbers)
#         result_1 += [((version_3_l2_star_discrepancy(point_set))**.5).real]
#         result_2 += [star_discrepancy_parallel(point_set)]

#     vector_results_1 += [result_1]
#     vector_results_2 += [result_2]
#     mid_time = time.perf_counter() - start_time
#     # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
#     arr_1 = np.array(vector_results_1)
#     # print(arr)
#     pd.DataFrame(arr_1).to_csv('L2 star Discrepancy Scrambled Sequence.csv')
#     arr_2 = np.array(vector_results_2)
#     # print(arr)
#     pd.DataFrame(arr_2).to_csv('Star Discrepancy Scrambled Sequence.csv')
#     print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
