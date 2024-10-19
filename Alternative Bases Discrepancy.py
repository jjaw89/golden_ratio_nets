from cmath import sqrt
from dataclasses import replace
from msilib.schema import Binary
from tkinter import S
from turtle import right
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
import timeit
import cupy as cp
from pathlib import Path
# plt.style.use('seaborn-whitegrid')


GOLDEN_RATIO = ((1+np.sqrt(5))/2).real


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
    return np.array([integer_to_decimal(n) for n in numbers])


@njit()
def convert_integer_points_to_decimals(points_n):
    return np.array([[integer_to_decimal(n[0]), integer_to_decimal(n[1])] for n in points_n], dtype='float')


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


def plot_points(points, x_partition, ypartition, size):
    x = points[:, 0]
    y = points[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=size)
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


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


@njit()
def generate_hammersley_numbers(num_digits):
    x_numbers = no_consecutive_ones_binary_flipped(num_digits)
    y_numbers = no_consecutive_ones_binary_2(num_digits)
    point_numbers = np.array([[x_numbers[n], y_numbers[n]]
                             for n in range(FIBONACCI[num_digits+2])], dtype='int64')
    return point_numbers


@njit(parallel=True)
def star_discrepancy_parallel_find_box(point_set):
    points = point_set[point_set[:, 0].argsort()]
    ones = np.ones((1, 2), dtype='float')
    points = np.append(ones, points, axis=0)
    num_points = points.shape[0]

    # y_values_gpu = cp.array([1, 1], dtype='int32')

    closed_disc_max = np.zeros(num_points, dtype='float')
    closed_right_side_max = np.zeros(num_points, dtype='float')
    closed_top_side_max = np.zeros(num_points, dtype='float')

    open_disc_max = np.zeros(num_points, dtype='float')
    open_right_side_max = np.zeros(num_points, dtype='float')
    open_top_side_max = np.zeros(num_points, dtype='float')

    for right_side_index in prange(1, num_points-1):
        y_coords_sorted = np.argsort(points[:right_side_index+1, 1])

        inner_closed_disc_max = 0
        inner_open_disc_max = 0
        inner_closed_right_side_max = 0
        inner_closed_top_side_max = 0
        inner_open_right_side_max = 0
        inner_open_top_side_max = 0
        for top_side_index in range(1, right_side_index):

            # print(
            #     f'right_side_index : {right_side_index}, top_side_index : {top_side_index} : {y_coords_sorted[top_side_index+1]}')

            if (top_side_index+1) / num_points - points[right_side_index, 0] * points[y_coords_sorted[top_side_index], 1] > inner_closed_disc_max:
                inner_closed_disc_max = (top_side_index+1) / num_points - \
                    points[right_side_index, 0] * \
                    points[y_coords_sorted[top_side_index],
                           1]
                inner_closed_right_side_max = points[right_side_index, 0]
                inner_closed_top_side_max = points[y_coords_sorted[top_side_index], 1]

            if -((top_side_index+1) / num_points - points[right_side_index + 1, 0] * points[y_coords_sorted[top_side_index+1], 1]) > inner_open_disc_max:
                inner_open_disc_max = -((top_side_index+1) / num_points -
                                        points[right_side_index + 1, 0] *
                                        points[y_coords_sorted[top_side_index+1],
                                               1])
                inner_open_right_side_max = points[right_side_index + 1, 0]
                inner_open_top_side_max = points[y_coords_sorted[top_side_index+1], 1]

        # top_side_index = right_side_index-1

        # if top_side_index / num_points - points[right_side_index, 0] * points[y_coords_sorted[top_side_index], 1] > inner_closed_disc_max:
        #     inner_closed_disc_max = top_side_index / num_points - \
        #         points[right_side_index, 0] * \
        #         points[y_coords_sorted[top_side_index],
        #                1]
        #     inner_closed_right_side_max = points[right_side_index, 0]
        #     inner_closed_top_side_max = points[y_coords_sorted[top_side_index], 1]

        # if -(top_side_index / num_points - points[right_side_index + 1, 0]) > inner_open_disc_max:
        #     inner_open_disc_max = -(top_side_index / num_points -
        #                             points[right_side_index + 1, 0])
        #     inner_open_right_side_max = points[right_side_index + 1, 0]
        #     inner_open_top_side_max = 1

        closed_disc_max[right_side_index] = inner_closed_disc_max
        closed_right_side_max[right_side_index] = inner_closed_right_side_max
        closed_top_side_max[right_side_index] = inner_closed_top_side_max

        open_disc_max[right_side_index] = inner_open_disc_max
        open_right_side_max[right_side_index] = inner_open_right_side_max
        open_top_side_max[right_side_index] = inner_open_top_side_max

    closed_disc_max_index = np.argmax(closed_disc_max)
    open_disc_max_index = np.argmax(open_disc_max)

    results = np.zeros((2, 3), dtype='float')
    results[0, 0] = closed_disc_max[closed_disc_max_index]
    results[0, 1] = closed_right_side_max[closed_disc_max_index]
    results[0, 2] = closed_top_side_max[closed_disc_max_index]

    results[1, 0] = open_disc_max[open_disc_max_index]
    results[1, 1] = open_right_side_max[open_disc_max_index]
    results[1, 2] = open_top_side_max[open_disc_max_index]

    return results


@njit(parallel=True)
def star_discrepancy_parallel_find_box_correct(point_set):
    points = point_set[point_set[:, 0].argsort()]
    x = points[:, 0]
    y = points[:, 1]
    n = points.shape[0]
    one = np.ones(1, dtype='float')
    zero = np.zeros(1, dtype='float')
    x = np.append(x, one, axis=0)
    x = np.append(zero, x, axis=0)

    closed_disc_max = np.zeros(n+2, dtype='float')
    closed_right_side_max = np.zeros(n+2, dtype='float')
    closed_top_side_max = np.zeros(n+2, dtype='float')

    open_disc_max = np.zeros(n+2, dtype='float')
    open_right_side_max = np.zeros(n+2, dtype='float')
    open_top_side_max = np.zeros(n+2, dtype='float')

    for j in prange(0, n+1):

        xi = np.sort(y[:j])
        xi = np.append(xi, one, axis=0)
        xi = np.append(zero, xi, axis=0)
        # print(xi)
        inner_closed_disc_max = 0
        inner_closed_right_side_max = 0
        inner_closed_top_side_max = 0

        inner_open_disc_max = 0
        inner_open_right_side_max = 0
        inner_open_top_side_max = 0

        for k in range(0, j+1):

            # print(
            #     f'right_side_index : {right_side_index}, top_side_index : {top_side_index} : {y_coords_sorted[top_side_index+1]}')

            if (k) / n - x[j] * xi[k] > inner_closed_disc_max:
                inner_closed_disc_max = (k) / n - x[j] * xi[k]
                inner_closed_right_side_max = x[j]
                inner_closed_top_side_max = xi[k]

            if (k) / n - x[j+1] * xi[k+1] > inner_closed_disc_max:
                inner_open_disc_max = (k) / n - x[j+1] * xi[k+1]
                inner_open_right_side_max = x[j+1]
                inner_open_top_side_max = xi[k+1]

        closed_disc_max[j] = inner_closed_disc_max
        closed_right_side_max[j] = inner_closed_right_side_max
        closed_top_side_max[j] = inner_closed_top_side_max

        open_disc_max[j] = inner_open_disc_max
        open_right_side_max[j] = inner_open_right_side_max
        open_top_side_max[j] = inner_open_top_side_max

    # print(closed_disc_max)
    closed_disc_max_index = np.argmax(closed_disc_max)
    open_disc_max_index = np.argmax(open_disc_max)

    results = np.zeros((2, 3), dtype='float')
    results[0, 0] = closed_disc_max[closed_disc_max_index]
    results[0, 1] = closed_right_side_max[closed_disc_max_index]
    results[0, 2] = closed_top_side_max[closed_disc_max_index]

    results[1, 0] = open_disc_max[open_disc_max_index]
    results[1, 1] = open_right_side_max[open_disc_max_index]
    results[1, 2] = open_top_side_max[open_disc_max_index]

    return results


@njit()
def star_discrepancy_parallel_find_box_correct_insert(point_set):
    points = point_set[point_set[:, 0].argsort()]
    x = points[:, 0]
    y = points[:, 1]
    n = points.shape[0]
    one = np.ones(1, dtype='float')
    zero = np.zeros(1, dtype='float')
    x = np.append(x, one, axis=0)
    x = np.append(zero, x, axis=0)

    closed_disc_max = np.zeros(n+2, dtype='float')
    closed_right_side_max = np.zeros(n+2, dtype='float')
    closed_top_side_max = np.zeros(n+2, dtype='float')

    open_disc_max = np.zeros(n+2, dtype='float')
    open_right_side_max = np.zeros(n+2, dtype='float')
    open_top_side_max = np.zeros(n+2, dtype='float')

    xi = np.ones(points.shape[0]+3, dtype='float')
    xi[0] = 0

    for j in range(0, n+1):

        # print(xi)

        inner_closed_disc_max = 0
        inner_closed_right_side_max = 0
        inner_closed_top_side_max = 0

        inner_open_disc_max = 0
        inner_open_right_side_max = 0
        inner_open_top_side_max = 0

        for k in range(0, j+1):

            # print(
            #     f'right_side_index : {right_side_index}, top_side_index : {top_side_index} : {y_coords_sorted[top_side_index+1]}')

            if (k) / n - x[j] * xi[k] > inner_closed_disc_max:
                inner_closed_disc_max = (k) / n - x[j] * xi[k]
                inner_closed_right_side_max = x[j]
                inner_closed_top_side_max = xi[k]

            if (k) / n - x[j+1] * xi[k+1] > inner_closed_disc_max:
                inner_open_disc_max = (k) / n - x[j+1] * xi[k+1]
                inner_open_right_side_max = x[j+1]
                inner_open_top_side_max = xi[k+1]

        closed_disc_max[j] = inner_closed_disc_max
        closed_right_side_max[j] = inner_closed_right_side_max
        closed_top_side_max[j] = inner_closed_top_side_max

        open_disc_max[j] = inner_open_disc_max
        open_right_side_max[j] = inner_open_right_side_max
        open_top_side_max[j] = inner_open_top_side_max

        insert_here = np.searchsorted(xi, y[j])
        xi[insert_here+1:j+3] = xi[insert_here:j+2]
        xi[insert_here] = y[j]
    # print(closed_disc_max)
    closed_disc_max_index = np.argmax(closed_disc_max)
    open_disc_max_index = np.argmax(open_disc_max)

    results = np.zeros((2, 3), dtype='float')
    results[0, 0] = closed_disc_max[closed_disc_max_index]
    results[0, 1] = closed_right_side_max[closed_disc_max_index]
    results[0, 2] = closed_top_side_max[closed_disc_max_index]

    results[1, 0] = open_disc_max[open_disc_max_index]
    results[1, 1] = open_right_side_max[open_disc_max_index]
    results[1, 2] = open_top_side_max[open_disc_max_index]

    return results


@njit(parallel=True)
def star_discrepancy_parallel_find_box_correct_insert_parallel(point_set):
    points = point_set[point_set[:, 0].argsort()]
    x = points[:, 0]
    y = points[:, 1]
    n = points.shape[0]
    one = np.ones(1, dtype='float')
    zero = np.zeros(1, dtype='float')
    x = np.append(x, one, axis=0)
    x = np.append(zero, x, axis=0)

    closed_disc_max = 0
    closed_right_side_max = 0
    closed_top_side_max = 0

    open_disc_max = 0
    open_right_side_max = 0
    open_top_side_max = 0

    xi = np.ones(points.shape[0]+4, dtype='float')
    xi[0] = 0
    inner_closed_disc_max = np.zeros((n+3,), dtype='float')

    inner_open_disc_max = np.zeros((n+3,), dtype='float')
    for j in range(0, n+1):

        # print(xi)
        for k in range(j+1):
            inner_closed_disc_max[k] = (k) / n - x[j] * xi[k]
            inner_open_disc_max[k] = (k) / n - x[j+1] * xi[k+1]

        # print(inner_closed_disc_max)
        closed_disc_max_index = np.argmax(inner_closed_disc_max[:j+1])
        open_disc_max_index = np.argmax(inner_open_disc_max[:j+1])

        if inner_closed_disc_max[closed_disc_max_index] > closed_disc_max:
            closed_disc_max = inner_closed_disc_max[closed_disc_max_index]
            closed_right_side_max = x[j]
            closed_top_side_max = xi[closed_disc_max_index]
        # print(closed_disc_max)
        if inner_open_disc_max[open_disc_max_index] > open_disc_max:
            open_disc_max = inner_open_disc_max[open_disc_max_index]
            open_right_side_max = x[j+1]
            open_top_side_max = xi[open_disc_max_index+1]

        insert_here = np.searchsorted(xi, y[j])
        xi[insert_here+1:j+3] = xi[insert_here:j+2]
        xi[insert_here] = y[j]
    # print(closed_disc_max)

    results = np.zeros((2, 3), dtype='float')
    results[0, 0] = closed_disc_max
    results[0, 1] = closed_right_side_max
    results[0, 2] = closed_top_side_max

    results[1, 0] = open_disc_max
    results[1, 1] = open_right_side_max
    results[1, 2] = open_top_side_max

    return results


@njit()
def star_discrepancy_largest_rest(point_set):
    n = point_set.shape[0]

    points = point_set[point_set[:, 0].argsort()]

    # Pad the x values
    x = points[:, 0]
    one = np.ones(1, dtype='float')
    zero = np.zeros(1, dtype='float')
    x = np.append(x, one, axis=0)
    x = np.append(zero, x, axis=0)

    # Pad y values
    y = points[:, 1]
    one = np.ones(1, dtype='float')
    zero = np.zeros(1, dtype='float')
    y = np.append(zero, y, axis=0)

    # create base case i=0 xi
    xi = np.ones(n+3, dtype='float')
    xi[0] = 0

    # Create vector to store results
    # results[0, 0] = closed_disc_max
    # results[0, 1] = closed_right_side_max
    # results[0, 2] = closed_top_side_max

    # results[1, 0] = open_disc_max
    # results[1, 1] = open_right_side_max
    # results[1, 2] = open_top_side_max
    results = np.zeros((2, 3), dtype='float')

    max_closed = 0
    max_closed_right = 0
    max_closed_top = 0

    max_open = x[1]
    max_open_right = x[1]
    max_open_top = 1
    # print(xi)
    for j in range(1, n+1):
        insert_here = np.searchsorted(xi, y[j], side='right')
        # print(insert_here)

        # print(xi[insert_here:j])
        # print(xi[insert_here+1:j+1])
        for k in range(j-insert_here+1):
            xi[j+1-k] = xi[j-k]
        # print(xi)
        # print(y[j])
        xi[insert_here] = y[j]
        # print(xi)
        # print()
        for k in range(0, j+1):
            if k/n - x[j] * xi[k] > max_closed:
                max_closed = k/n - x[j]*xi[k]
                max_closed_right = x[j]
                max_closed_top = xi[k]
            # if x[j+1] * xi[k+1] - k/n > max_open:
            #     max_open = k/n - x[j+1]*xi[k+1]
            #     max_open_right = x[j+1]
            #     max_open_top = xi[j+1]

    results[0, 0] = max_closed
    results[0, 1] = max_closed_right
    results[0, 2] = max_closed_top
    results[1, 0] = max_open
    results[1, 1] = max_open_right
    results[1, 2] = max_open_top

    return results


@njit()
def star_discrepancy_largest_rest_2(point_set):
    n = point_set.shape[0]

    points = point_set[point_set[:, 0].argsort()]

    # Pad the x values
    x = points[:, 0]
    one = np.ones(1, dtype='float')
    zero = np.zeros(1, dtype='float')
    x = np.append(x, one, axis=0)
    x = np.append(zero, x, axis=0)

    # Pad y values
    y = points[:, 1]
    one = np.ones(1, dtype='float')
    zero = np.zeros(1, dtype='float')
    y = np.append(zero, y, axis=0)

    # create base case i=0 xi
    xi = np.ones(n+3, dtype='float')
    xi[0] = 0

    # Create vector to store results
    # results[0, 0] = closed_disc_max
    # results[0, 1] = closed_right_side_max
    # results[0, 2] = closed_top_side_max

    # results[1, 0] = open_disc_max
    # results[1, 1] = open_right_side_max
    # results[1, 2] = open_top_side_max
    results = np.zeros((2, 3), dtype='float')

    max_closed = 0
    max_closed_right = 0
    max_closed_top = 0

    max_open = x[1]
    max_open_right = x[1]
    max_open_top = 1
    # print(xi)
    for j in range(1, n+1):
        insert_here = np.searchsorted(xi, y[j], side='right')
        # print(insert_here)

        # print(xi[insert_here:j])
        # print(xi[insert_here+1:j+1])
        for k in range(j-insert_here+1):
            xi[j+1-k] = xi[j-k]
        # print(xi)
        # print(y[j])
        xi[insert_here] = y[j]
        # print(xi)
        # print()
        for k in range(0, j+1):
            if k/n - x[j] * xi[k] > max_closed:
                max_closed = k/n - x[j]*xi[k]
                max_closed_right = x[j]
                max_closed_top = xi[k]
            if x[j+1] * xi[k+1] - k/n > max_open:
                max_open = k/n - x[j+1]*xi[k+1]
                max_open_right = x[j+1]
                max_open_top = xi[j+1]

    results[0, 0] = max_closed
    results[0, 1] = max_closed_right
    results[0, 2] = max_closed_top
    results[1, 0] = max_open
    results[1, 1] = max_open_right
    results[1, 2] = max_open_top

    return results


path = Path('~\Dropbox\PythonProjects').expanduser()
path.mkdir(parents=True, exist_ok=True)
L2_star_discrepancy = np.zeros((30, 3), dtype='float')
# for num_digits in range(1, 30):

#     points = convert_integer_points_to_decimals(
#         generate_hammersley_numbers(num_digits))
#     # ones = np.ones((1, 2), dtype='float')
#     # points = np.append(points, ones, axis=0)
#     # print(points)

#     # start_time = time.perf_counter()
#     # result = star_discrepancy_parallel_find_box_correct(points)
#     # total_time_1 = time.perf_counter() - start_time

#     # # print(num_digits)
#     # if result[0, 0] > result[1, 0]:
#     #     print(
#     #         f'{num_digits} , c, {result[0,0]}, {result[0,1]} , {result[0,2]}')
#     # else:
#     #     print(
#     #         f'{num_digits} , o, {result[0,0]}, {result[0,1]} , {result[0,2]}')

#     start_time = time.perf_counter()
#     result = star_discrepancy_parallel_find_box_correct_insert(points)
#     total_time_2 = time.perf_counter() - start_time

#     L2_star_discrepancy[num_digits] = result[0, :]
#     np.save(path/'L2_star_discrepancy', star_discrepancy)
#     if result[0, 0] > result[1, 0]:
#         print(
#             f'{num_digits} , c, {result[0,0]}, {result[0,1]} , {result[0,2]}')
#     else:
#         print(
#             f'{num_digits} , o, {result[0,0]}, {result[0,1]} , {result[0,2]}')
#     # print(f'{num_digits} : {total_time_1}')
#     # print(f'{num_digits} :  {total_time_2}')
#     # print(result)
#     print()


points = np.array([[0, 0], [.5, .5], [.25, .25]], dtype='float')


def l2_star_discrepancy_cuda_first_part(points):
    cuda_points = cp.asarray(points, dtype='float')

    first_term = cp.sum(cp.product(1-cp.square(cuda_points), axis=1) - 4/9)
    return first_term


def l2_star_discrepancy_cuda_second_part(points):
    cuda_points_x = cp.asarray(1-points[:, 0], dtype='float')
    cuda_points_y = cp.asarray(1-points[:, 1], dtype='float')
    sum = 0.
    for n in range(1, cuda_points_x.shape[0]):
        sum += cp.sum(cp.minimum(cuda_points_x, cp.roll(cuda_points_x, n))
                      * cp.minimum(cuda_points_y, cp.roll(cuda_points_y, n)) - 1/9)
    return sum


def l2_star_discrepancy_cuda_second_part_2(points):
    cuda_points_x = cp.asarray(1-points[:, 0], dtype='float')
    cuda_points_y = cp.asarray(1-points[:, 1], dtype='float')
    sum = 0.
    if (cuda_points_x.shape[0]) % 2 == 0:
        for n in range(1, (cuda_points_x.shape[0]) // 2):
            sum += cp.sum(cp.minimum(cuda_points_x, cp.roll(cuda_points_x, n))
                          * cp.minimum(cuda_points_y, cp.roll(cuda_points_y, n)) - 1/9)
        sum *= 2
        sum += cp.sum(cp.minimum(cuda_points_x, cp.roll(cuda_points_x, (cuda_points_x.shape[0]) // 2))
                      * cp.minimum(cuda_points_y, cp.roll(cuda_points_y, (cuda_points_x.shape[0]) // 2)) - 1/9)

    else:
        for n in range(1, cuda_points_x.shape[0] // 2+1):
            sum += cp.sum(cp.minimum(cuda_points_x, cp.roll(cuda_points_x, n))
                          * cp.minimum(cuda_points_y, cp.roll(cuda_points_y, n)) - 1/9)
        sum *= 2
    return sum


def l2_star_discrepancy_cuda_third_part(points):
    cuda_points = cp.asarray(points, dtype='float')
    return cp.sum(cp.product(1-cuda_points, axis=1)-1/4)


def l2_star_discrepancy_cuda(points):
    return 1/points.shape[0]*(1/4-1/9) - 1/(2*points.shape[0]) * l2_star_discrepancy_cuda_first_part(points)+1/(points.shape[0])**2*(l2_star_discrepancy_cuda_second_part_2(points)+l2_star_discrepancy_cuda_third_part(points))


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


num_digits = 1
point_numbers = np.array(
    [[0, 0], [1, 1]], dtype='int64')

# num_digits = 2
# point_numbers = np.array(
#     [[0, 0], [1, 1], [2, 2]], dtype='int64')

# print(no_consecutive_ones_binary_2(4))
# print(no_consecutive_ones_binary_flipped(2))


# extend_points(point_numbers, num_digits)


total_start_time = time.perf_counter()
for _ in range(31):  # 26 for starting with 2
    start_time = time.perf_counter()
    point_numbers = extend_points(point_numbers, num_digits)
    num_digits += 1
    end_time = time.perf_counter()
    print(
        f"Number of digits : {num_digits}, Time : {end_time-start_time:0.6f}")
total_end_time = time.perf_counter()
print(
    f"Total_time : {total_end_time-total_start_time:0.6f}")

points = convert_integer_points_to_decimals(point_numbers)


for m in range(3):
    start_time = time.perf_counter()
    l2 = l2_star_discrepancy_cuda(points[:FIBONACCI[m+2]])
    star = star_discrepancy_largest_rest(points[:FIBONACCI[m+2]])[0, 0]
    end_time = time.perf_counter()
    print(f'{start_time-end_time}, {m}, {star}, {l2}')


# for m in range(3, 6):
#     plot_points(points[:FIBONACCI[m+2]], 0, 0, 10)
