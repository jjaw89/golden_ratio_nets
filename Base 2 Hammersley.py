
from cmath import sqrt
from tkinter import N
from xml.dom import WrongDocumentErr
import numpy as np
import time
import sys
import cupy as cp
import math
import scipy.special
from numba import jit, njit, vectorize, prange
import concurrent.futures
import shutil
import pandas as pd

import matplotlib.pyplot as plt
from sympy import GoldenRatio, plot
plt.style.use('seaborn-whitegrid')


def binary_vectors(length):
    end_in_zero = np.zeros((1, 1), dtype='b')
    end_in_one = np.ones((1, 1), dtype='b')

    vectors = np.append(end_in_one, end_in_zero, axis=0)

    for i in range(length-1):
        vectors_zero = np.append(vectors,
                                 np.zeros(
                                         (vectors.shape[0], 1), dtype='b'),
                                 axis=1)
        vectors_one = np.append(vectors,
                                np.ones(
                                    (vectors.shape[0], 1), dtype='b'),
                                axis=1)
        vectors = np.append(vectors_one, vectors_zero, axis=0)

    return vectors[np.lexsort(np.rot90(vectors))]


def base_2_multiplied_2d(num_digits):
    multiplied_matrix = np.array(
        [[2 ** (-1), 2 ** (-num_digits)]])
    for i in range(2, num_digits+1):
        multiplied_matrix = np.append(multiplied_matrix,
                                      [[2 ** (-i), 2 ** (i-num_digits-1)]], axis=0)
    return multiplied_matrix


def base_2_multiplied_1d(num_digits):
    multiplied_matrix = np.array(
        [[2 ** (-1)]])
    for i in range(2, num_digits+1):
        multiplied_matrix = np.append(multiplied_matrix,
                                      [[2 ** (-i)]], axis=0)
    return multiplied_matrix


def partition(num_digits):
    if num_digits == 0:
        return np.array([0, 1]).real
    numbers = np.matmul(binary_vectors(
        num_digits), base_2_multiplied_1d(num_digits))
    numbers = np.append(numbers[numbers[:, 0].argsort()], [[1]], axis=0)
    return numbers.real[:, 0]


def points_sorted(num_digits):
    points = np.matmul(binary_vectors(
        num_digits), base_2_multiplied_2d(num_digits))
    points = points[points[:, 0].argsort()]
    return points.real


def points(num_digits):
    points = np.matmul(binary_vectors(num_digits),
                       base_2_multiplied_2d(num_digits))
    return points.real


def points_shuffled(num_digits):
    points = np.matmul(binary_vectors(
        num_digits), base_2_multiplied_2d(num_digits))
    np.random.shuffle(points)
    return points


def plot_hammersley(num_digits, x_partition, ypartition):
    hammersley = points(num_digits)
    x = hammersley[:, 0]
    y = hammersley[:, 1]
    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.',  markersize=10)
    plt.title(f'Base 2 Hammersley set with {num_digits} digits')
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


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
def star_discrepancy_parallel_find_box(point_set):
    points = point_set[point_set[:, 0].argsort()]
    num_points = points.shape[0]

    closed_disc_max = np.zeros(num_points, dtype='float')
    closed_right_side_max = np.zeros(num_points, dtype='float')
    closed_top_side_max = np.zeros(num_points, dtype='float')

    open_disc_max = np.zeros(num_points, dtype='float')
    open_right_side_max = np.zeros(num_points, dtype='float')
    open_top_side_max = np.zeros(num_points, dtype='float')

    for right_side_index in range(0, num_points):
        points_temp = points[:right_side_index]
        ones = np.ones((1, 2), dtype='float')
        points_temp = np.append(points_temp, ones, axis=0)
        y_coords_sorted = np.argsort(points_temp[:, 1])
        print('*')
        print(y_coords_sorted)
        print()
        inner_closed_disc_max = 0
        inner_open_disc_max = 0
        inner_closed_right_side_max = 0
        inner_closed_top_side_max = 0
        inner_open_right_side_max = 0
        inner_open_top_side_max = 0
        for top_side_index in range(0, right_side_index):

            # print(
            #     f'right_side_index : {right_side_index}, top_side_index : {top_side_index} : {y_coords_sorted[top_side_index+1]}')

            if (top_side_index+1) / num_points - points_temp[right_side_index, 0] * points_temp[:right_side_index+1][y_coords_sorted[top_side_index], 1] > inner_closed_disc_max:
                inner_closed_disc_max = (top_side_index+1) / num_points - \
                    points_temp[right_side_index, 0] * \
                    points_temp[:right_side_index+1][y_coords_sorted[top_side_index],
                                                     1]
                inner_closed_right_side_max = points_temp[right_side_index, 0]
                inner_closed_top_side_max = points_temp[y_coords_sorted[top_side_index], 1]

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

    print(closed_disc_max)
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
def star_discrepancy_parallel_find_box(point_set):
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


# for num_digits in range(1, 19):

#     points = points_sorted(num_digits)
#     # ones = np.ones((1, 2), dtype='float')
#     # points = np.append(points, ones, axis=0)
#     # print(points)
#     result = star_discrepancy_parallel_find_box(points)
#     # print(num_digits)
#     if result[0, 0] > result[1, 0]:
#         print(
#             f'{num_digits} , c, {result[0,0]}, {result[0,1]} , {result[0,2]}')
#     else:
#         print(
#             f'{num_digits} , o, {result[0,0]}, {result[0,1]} , {result[0,2]}')

#     # n = 2*num_digits
#     # disc = 4 * n / (4 * 2**n*(3)) + 2**-n * (5/4+(4+3) /
#     #                                          (4*3**2)) - 1/(4 * 2 ** (2 * n))*(1+(4+3)/(9))
#     # print(disc)
    # print(result)


# num_digits = 1
# point_set = points(num_digits)


# vector_results = []
# for num_digits in range(21):
#     start_time = time.perf_counter()
#     point_set = points_shuffled(num_digits)
#     vector_results += [[point_set.shape[0],
#                         version_3_l2_star_discrepancy(point_set)]]
#     mid_time = time.perf_counter() - start_time
#     # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
#     arr = np.array(vector_results)
#     # print(arr)
#     pd.DataFrame(arr).to_csv('results base phi.csv')
#     print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
# print(arr)

# start_time = time.perf_counter()
# num_digits = 16
# point_set = points_sorted(num_digits)
# print(star_discrepancy(point_set))
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Star discrepancy time : {time_difference :0.6f}")

# start_time = time.perf_counter()
# num_digits = 17
# point_set = points_sorted(num_digits)
# print(star_discrepancy(point_set))
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Star discrepancy time : {time_difference :0.6f}")

# vector_results = []
# for num_digits in range(16):
#     start_time = time.perf_counter()
#     point_set = points_sorted(num_digits)
#     vector_results += [[point_set.shape[0],
#                         star_discrepancy(point_set)]]
#     mid_time = time.perf_counter() - start_time
#     # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
#     arr = np.array(vector_results)
#     # print(arr)
#     pd.DataFrame(arr).to_csv('star discrepancy base 2.csv')
#     print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
# print(arr)


def plot_hammersley(num_digits, x_partition, ypartition):
    hammersley = points(num_digits)
    x = hammersley[:, 0]
    y = hammersley[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.',  markersize=5)
    # plt.title(f'Base 2 Hammersley set with {num_digits} digits')
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


plot_hammersley(13, 0, 0)
