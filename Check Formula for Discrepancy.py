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
star_discrepancy = np.zeros((30, 3), dtype='float')
for num_digits in range(1, 30):

    points = convert_integer_points_to_decimals(
        generate_hammersley_numbers(num_digits))
    # ones = np.ones((1, 2), dtype='float')
    # points = np.append(points, ones, axis=0)
    # print(points)

    # start_time = time.perf_counter()
    # result = star_discrepancy_parallel_find_box_correct(points)
    # total_time_1 = time.perf_counter() - start_time

    # # print(num_digits)
    # if result[0, 0] > result[1, 0]:
    #     print(
    #         f'{num_digits} , c, {result[0,0]}, {result[0,1]} , {result[0,2]}')
    # else:
    #     print(
    #         f'{num_digits} , o, {result[0,0]}, {result[0,1]} , {result[0,2]}')

    start_time = time.perf_counter()
    result = star_discrepancy_parallel_find_box_correct_insert(points)
    total_time_2 = time.perf_counter() - start_time

    star_discrepancy[num_digits] = result[0, :]
    np.save(path/'star_discrepancy', star_discrepancy)
    if result[0, 0] > result[1, 0]:
        print(
            f'{num_digits} , c, {result[0,0]}, {result[0,1]} , {result[0,2]}')
    else:
        print(
            f'{num_digits} , o, {result[0,0]}, {result[0,1]} , {result[0,2]}')
    # print(f'{num_digits} : {total_time_1}')
    # print(f'{num_digits} :  {total_time_2}')
    # print(result)
    print()


# array = np.arange(6)
# print(np.searchsorted(array, 4))
# print(np.insert(array, 3, 0))

# num_digits = 3
# points = convert_integer_points_to_decimals(
#     generate_hammersley_numbers(num_digits))
# print(points)
# start_time = time.perf_counter()
# star_discrepancy_largest_rest(points)
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 1 {total_time}')
# print()
# start_time = time.perf_counter()
# star_discrepancy_parallel_find_box_correct_insert(points)
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 2 {total_time}')
# print()

# num_digits = 5
# points = convert_integer_points_to_decimals(
#     generate_hammersley_numbers(num_digits))
# start_time = time.perf_counter()
# print(star_discrepancy_largest_rest(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 1 {total_time}')
# start_time = time.perf_counter()
# print(star_discrepancy_parallel_find_box_correct_insert(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 2 {total_time}')
# print()

# num_digits = 10
# points = convert_integer_points_to_decimals(
#     generate_hammersley_numbers(num_digits))
# start_time = time.perf_counter()
# print(star_discrepancy_largest_rest(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 1 {total_time}')
# start_time = time.perf_counter()
# print(star_discrepancy_parallel_find_box_correct_insert(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 2 {total_time}')
# print()

# num_digits = 15
# points = convert_integer_points_to_decimals(
#     generate_hammersley_numbers(num_digits))
# start_time = time.perf_counter()
# print(star_discrepancy_largest_rest(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 1 {total_time}')
# start_time = time.perf_counter()
# print(star_discrepancy_parallel_find_box_correct_insert(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 2 {total_time}')
# print()

# num_digits = 20
# points = convert_integer_points_to_decimals(
#     generate_hammersley_numbers(num_digits))
# start_time = time.perf_counter()
# print(star_discrepancy_largest_rest(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 1 {total_time}')
# start_time = time.perf_counter()
# print(star_discrepancy_parallel_find_box_correct_insert(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 2 {total_time}')
# print()

# num_digits = 25
# points = convert_integer_points_to_decimals(
#     generate_hammersley_numbers(num_digits))
# start_time = time.perf_counter()
# print(star_discrepancy_largest_rest(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 1 {total_time}')
# start_time = time.perf_counter()
# print(star_discrepancy_parallel_find_box_correct_insert(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 2 {total_time}')
# print()

# num_digits = 30
# points = convert_integer_points_to_decimals(
#     generate_hammersley_numbers(num_digits))
# start_time = time.perf_counter()
# print(star_discrepancy_largest_rest(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 1 {total_time}')
# start_time = time.perf_counter()
# print(star_discrepancy_parallel_find_box_correct_insert(points))
# total_time = time.perf_counter() - start_time
# print(f'{num_digits} : 2 {total_time}')
# print()

# num_digits = 20
# numbers = no_consecutive_ones_binary_2(num_digits)
# vdC = convert_integers_to_decimals(numbers)
# value_arr = np.array([(np.sum(1/GOLDEN_RATIO-vdC[:n])) / (math.log(n)/math.log(GOLDEN_RATIO))
#                      for n in range(2, FIBONACCI[num_digits+2])])
# value_arr_2 = np.zeros(FIBONACCI[num_digits+2])
# value_arr_2[2:] += value_arr
# # log_term = math.log2(n)/math.log2(GOLDEN_RATIO)
# # x = (np.sum(1/2-vdC[:n])) / (log_term)
# # if x > .125:
# #     print(f"{n} fail")
# # print(f"{n} , {x}")
# # print(np.sum((vdC[0:n])))
# print(value_arr_2[:13])
# maximums = np.zeros((num_digits, 2), dtype='int32')
# maximums_2 = np.zeros((num_digits, 2), dtype='float')
# for m in range(1, num_digits):
#     n = np.argmax(value_arr_2[FIBONACCI[m+1]:FIBONACCI[m+2]])
#     n += FIBONACCI[m+1]
#     maximums[m, 0] = n
#     maximums[m, 1] = n-maximums[m-1, 0]
#     maximums_2[m, 0] = value_arr_2[n]
#     maximums_2[m, 1] = value_arr_2[n]-maximums_2[m-1, 0]
#     # maximums[2, m] = value_arr_2[n]

# print(maximums)
# print(maximums_2)
# base_f_expansion = np.zeros((num_digits, num_digits), dtype='int32')
# for n in range(num_digits):
#     number = maximums[n, 0]
#     digit_counter = num_digits+1

#     for m in range(num_digits):
#         digit = number // FIBONACCI[digit_counter]
#         # print(digit)
#         base_f_expansion[n, num_digits-1 - m] = digit
#         number -= digit*FIBONACCI[digit_counter]
#         digit_counter -= 1
# print(base_f_expansion)

# # base_phi_expansion = np.zeros((num_digits,num_digits),dtype = 'int32')
# # for n in range(num_digits):
# #     number = maximums_2[n]
# #     digit_counter = num_digits
# #     for m in range(num_digits):
# #         digit = number // GOLDEN_RATIO**(digit_counter)
# #         # print(digit)
# #         base_phi_expansion[n,m]  = digit
# #         number -= digit*GOLDEN_RATIO**(digit_counter)
# #         digit_counter -=1
# # print(base_phi_expansion)
# print(GOLDEN_RATIO)
# for n in range(num_digits):
#     print(value_arr_2[FIBONACCI[n]])


# A_m_star = np.zeros(num_digits)
# A_m_star = np.copy(A_m)

# add_term = np.array([A_m[n]-A_m[n-1]-A_m[n-2] for n in range(2, num_digits)])
# print(add_term)

# formula_add_term = np.array([FIBONACCI[n]*GOLDEN_RATIO**(-n)
#                              for n in range(2, num_digits)])
# # print(add_term-formula_add_term)

# for n in range(2, num_digits):
#     A_m_star[n] = A_m_star[n-1]+A_m_star[n-2] + \
#         FIBONACCI[n]*GOLDEN_RATIO**(-n)

# for n in range(2, num_digits):
#     A_m_star[n] = A_m_star[n-1]+A_m_star[n-2] + \
#         (1-(-1)**n * GOLDEN_RATIO**(-2*n))/math.sqrt(5).real
# print(A_m_star)

# B_m = np.zeros(num_digits)
# B_m[1] = GOLDEN_RATIO**(-1)
# B_m[2] = GOLDEN_RATIO**(-1)+1/math.sqrt(5).real
# for n in range(3, num_digits):
#     B_m[n] = 2*A_m_star[n-1]-A_m_star[n-2]

# print(A_m-B_m)
# print(A_m)
# x = sum_arr[FIBONACCI[m+2]:FIBONACCI[m+3]]
# y = (A_m[m]+sum_arr[0:FIBONACCI[m+1]]+GOLDEN_RATIO **
#      (-m-1)*range(1, FIBONACCI[m+1]+1))
# # print(sum_arr[1:FIBONACCI[m+1]+1])
# print(x-y)
# difference = np.array([(A_m[n]-A_m_star[n])
#                       for n in range(num_digits)])
# print(difference)


# num_digits = 20
# numbers = no_consecutive_ones_binary_2(num_digits)
# vdC = convert_integers_to_decimals(numbers)
# sum_arr = np.array([np.sum(1/2 - vdC[:n]) / math.log(n)/math.log(GOLDEN_RATIO)
#                     for n in range(2, FIBONACCI[num_digits+2])])
# # print(FIBONACCI[3:num_digits+2])
# print(sum_arr[FIBONACCI[3:num_digits+2]-2])
# print(sum_arr[:10])
# print(FIBONACCI[2:num_digits+2]-1)
# print(sum_arr[0])
# print(sum_arr[1])
# print(sum_arr[2])
# A_m = np.array([sum_arr[FIBONACCI[n]-1]
#                for n in range(2, num_digits+2)], dtype='float')


num_digits = 15
numbers = no_consecutive_ones_binary_2(num_digits)
vdC = convert_integers_to_decimals(numbers)
sum_arr = np.array([np.sum(1/2 - vdC[:n+1])
                    for n in range(FIBONACCI[num_digits+2])])

# print(sum_arr[:10])
# print(FIBONACCI[2:num_digits+2]-1)
# print(sum_arr[0])
# print(sum_arr[1])
# print(sum_arr[2])
A_m = np.array([sum_arr[FIBONACCI[n]-1]
               for n in range(2, num_digits+2)], dtype='float')
# print(A_m)

# C = np.array([(FIBONACCI[m+2]*FIBONACCI[m+2]-1) /
# #              (GOLDEN_RATIO**m * 2) for m in range(num_digits)])*GOLDEN_RATIO**(-2)*math.sqrt(5).real
# difference = A_m
# special_value = GOLDEN_RATIO**(-2)/(1-GOLDEN_RATIO**(-4))
# difference[0] = special_value
# print(difference)
# base_phi_expansion = np.zeros((num_digits, 30), dtype='int32')
# for n in range(num_digits):
#     number = abs(difference[n])

#     for m in range(30):
#         digit = math.floor(number * GOLDEN_RATIO**(m))
#         # print(digit)
#         base_phi_expansion[n, m] = digit
#         number -= digit*GOLDEN_RATIO**(-m)
# print(base_phi_expansion)


def solved_recurrence(m):
    return .5*GOLDEN_RATIO**(-2)*(-GOLDEN_RATIO)**(-m)-.1*(1+GOLDEN_RATIO**(-2))*(-GOLDEN_RATIO**(-2))**(m)+1/math.sqrt(5).real


# solved = np.array([solved_recurrence(m)
#                   for m in range(num_digits)], dtype='float')
# # print(solved)
# print(A_m-solved)
# num_digits = 20

# limit_value = GOLDEN_RATIO**(-2) * 1 / (1-GOLDEN_RATIO**(-4))
# print(limit_value)

# A_m = np.array([GOLDEN_RATIO**(m) for m in range(num_digits)])
# A_m_star = np.zeros(num_digits)

# A_m_star[0] = GOLDEN_RATIO**(4) * limit_value
# A_m_star[1] = GOLDEN_RATIO**(5) * limit_value

# for n in range(2, num_digits):
#     A_m_star[n] = A_m_star[n-1]+A_m_star[n-2] + \
#         FIBONACCI[n]*(GOLDEN_RATIO)**(-n)


# print(A_m_star / A_m)


# def whole_numbers_tilde(num_digits):
#     integers = no_consecutive_ones_binary_2(num_digits)

#     whole_numbers = np.zeros(FIBONACCI[num_digits+2], dtype='float')
#     psy = 1-GOLDEN_RATIO
#     for m in range(num_digits):
#         digit = integers % 2
#         whole_numbers += digit * psy**m
#         integers //= 2
#     return whole_numbers


# num_digits = 20
# sums = np.array([np.sum(whole_numbers_tilde(m))
#                 for m in range(num_digits)], dtype='float')
# special_number = GOLDEN_RATIO**(-2)/(1-GOLDEN_RATIO**(-4))
# upper_bounds = special_number * GOLDEN_RATIO**range(num_digits)

# base_phi_expansion = np.zeros((num_digits, num_digits), dtype='int32')
# for n in range(num_digits):
#     number = sums [n]
#     digit_counter = num_digits
#     for m in range(num_digits):
#         digit = number // GOLDEN_RATIO**(digit_counter)
#         # print(digit)
#         base_phi_expansion[n, m] = digit
#         number -= digit*GOLDEN_RATIO**(digit_counter)
#         digit_counter -= 1
# print(base_phi_expansion)
# print(sums)
# print(upper_bounds)
# print(upper_bounds - sums)
# @njit
# def sum_point_products(points):


#     return np.sum(np.prod(1-points, axis=1))


# points = convert_integer_points_to_decimals(
#     generate_hammersley_numbers(5))
# print(points)

# print(np.array([(1-points[n, 0])*(1-points[n, 1])
#                 for n in range(points.shape[0])]))
# points = 1-points
# print(np.prod(points, axis=1))
# for m in range(2, 20):
#     points = convert_integer_points_to_decimals(
#         generate_hammersley_numbers(m))
#     sum = np.sum(np.prod(1-points, axis=1))
#     print((sum / points.shape[0]-.25)*(points.shape[0]/m))
