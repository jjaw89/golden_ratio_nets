from scipy.stats import qmc
from sympy import GoldenRatio, plot
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import concurrent.futures
from numba import jit, njit, vectorize, prange
import scipy.special
import math
import cupy as cp
import sys
import time
import numpy as np
from xml.dom import WrongDocumentErr
from cmath import sqrt

plt.style.use('seaborn-whitegrid')

GOLDEN_RATIO = ((1+np.sqrt(5))/2).real


def plot_points(points, a, b, x_partition, ypartition, size):
    x = points[:, 0]
    y = points[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=size)
    plt.grid()
    # plt.xticks(ticks=increasing_whole_numbers_values_2(
    #     x_partition, a, b), labels=[])
    # plt.yticks(ticks=increasing_whole_numbers_values_2(
    #     ypartition, a, b), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def no_consecutive_ones(length):
    end_in_zero = np.zeros((1, 1), dtype=int)
    end_in_one = np.ones((1, 1), dtype=int)

    for i in range(length-1):
        end_in_zero_zero = np.append(end_in_zero,
                                     np.zeros(
                                         (end_in_zero.shape[0], 1), dtype=int),
                                     axis=1)
        end_in_zero_one = np.append(end_in_zero,
                                    np.ones(
                                        (end_in_zero.shape[0], 1), dtype=int),
                                    axis=1)
        end_in_one_zero = np.append(end_in_one,
                                    np.zeros(
                                        (end_in_one.shape[0], 1), dtype=int),
                                    axis=1)
        end_in_zero = np.append(end_in_zero_zero, end_in_one_zero, axis=0)
        end_in_one = end_in_zero_one

    return np.append(end_in_zero, end_in_one, axis=0)


def no_consecutive_ones_b(length):
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

    return np.append(end_in_zero, end_in_one, axis=0)


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
    numbers = np.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_1d(num_digits))
    numbers = np.append(numbers[numbers[:, 0].argsort()], [[1]], axis=0)
    return numbers[:, 0]


def points_sorted(num_digits):
    points = np.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_2d(num_digits))
    points = points[points[:, 0].argsort()]
    return points


def points(num_digits):
    points = np.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_2d(num_digits))
    return points


def points_shuffled(num_digits):
    points = np.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_2d(num_digits))
    np.random.shuffle(points)
    return points


def plot_hammersley(num_digits, x_partition, ypartition):
    hammersley = points(num_digits)
    x = hammersley[:, 0]
    y = hammersley[:, 1]
    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=10)
    plt.title(f'Base golden ratio Hammersley set with {num_digits} digits')
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def kronecker(num_digits):
    vectors = no_consecutive_ones_b(num_digits)
    print(vectors)
    vectors = vectors[np.lexsort(np.rot90(vectors))]
    print(vectors)
    print()
    x_coords = np.matmul(vectors, np.flip(
        golden_multiplied_1d(num_digits))).real
    print(x_coords)
    y_coords = np.array([[0.]])
    generator = GOLDEN_RATIO - 1
    for n in range(1, x_coords.size):
        y_coords = np.append(
            y_coords, [[generator * n - math.floor(generator * n)]], axis=0)
    print(y_coords)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x_coords[:, 0], y_coords[:, 0], 'o')
    plt.title(f'Base golden ratio Hammersley set with {num_digits} digits')
    plt.grid()
    plt.xticks(ticks=partition(num_digits), labels=[])
    plt.yticks(ticks=y_coords[:, 0], labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()
    return


# @njit(parallel=True)
# def second_part(arr):
#     var_sum = 0
#     for x in prange(arr.shape[0]):
#         for y in prange(arr.shape[0]):
#             var_sum += (1-max(arr[x, 0], arr[y, 0])) * \
#                 (1-max(arr[x, 1], arr[y, 1]))
#     return var_sum


# def first_part(arr):
#     var_sum = 0
#     # print()
#     for x in range(arr.shape[0]):
#         var_sum += (1-arr[x, 0]**2)*(1-arr[x, 1]**2)
#     return var_sum

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
# num_digits = 5
# point_set = points_sorted(num_digits)
# start_time = time.perf_counter()
# print(point_set.shape[0])
# print(version_3_second_part_1(point_set))
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Star discrepancy time : {time_difference :0.6f}")

# num_digits = 23
# point_set = points_sorted(num_digits)
# start_time = time.perf_counter()
# print(point_set.shape[0])
# print(version_3_second_part_1(point_set))
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Star discrepancy time : {time_difference :0.6f}")

# num_digits = 1
# point_set = points(num_digits)


sampler = qmc.Sobol(d=2, scramble=True)
sample = sampler.random_base2(13)
point_set_all = np.array(sample)
print(point_set_all.shape)

plot_points(point_set_all, 1, 1, 1, 1, 5)
# vector_results = []
# for num_digits in range(1, 20):
#     start_time = time.perf_counter()
#     point_set = np.copy(point_set_all[:2**num_digits, :])
#     np.random.shuffle(point_set)
#     vector_results += [[point_set.shape[0],
#                         version_3_l2_star_discrepancy(point_set)]]
#     mid_time = time.perf_counter() - start_time
#     # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
#     arr = np.array(vector_results)
#     # print(arr)
#     pd.DataFrame(arr).to_csv('L2 star Discrepancy Sobol.csv')
#     print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
# print(arr)


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
# vector_results = []
# for num_digits in range(1, 20):
#     start_time = time.perf_counter()
#     point_set = 1-np.copy(point_set_all[:2**num_digits, :])
#     np.random.shuffle(point_set)
#     vector_results += [[point_set.shape[0],
#                         version_3_l2_star_discrepancy(point_set)]]
#     mid_time = time.perf_counter() - start_time
#     # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
#     arr = np.array(vector_results)
#     # print(arr)
#     pd.DataFrame(arr).to_csv('L2 star discrepancy 1- Sobol.csv')
#     print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
# print(arr)


# vector_results = []
# for n in range(16):
#     start_time = time.perf_counter()
#     point_set = np.copy(point_set_all[:2**n, :])
#     point_set = point_set[point_set[:, 0].argsort()]
#     vector_results += [[point_set.shape[0],
#                         star_discrepancy_parallel_find_box_correct(point_set)[0, 0]]]
#     mid_time = time.perf_counter() - start_time
#     # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
#     arr = np.array(vector_results)
#     # print(arr)
#     pd.DataFrame(arr).to_csv('Discrepancy Sobol.csv')
#     print(f"num digits : {n}, time so far : {mid_time:0.6f}")
# print(arr)

# point_set = points_sorted(5)
# point_set = 1-point_set
# point_set = np.flip(point_set, axis=0)
# print(point_set)


# start_time = time.perf_counter()
# num_digits = 30
# point_set = points_shuffled(num_digits)
# print([num_digits, point_set.shape[0],
#        version_3_l2_star_discrepancy(point_set)])
# mid_time = time.perf_counter() - start_time
# print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")


# start_time = time.perf_counter()
# num_digits = 20
# point_set = points(num_digits)
# print(
#     f"Number of digits : {num_digits}, number of points : {point_set.shape[0]}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Point Creation Time : {time_difference :0.6f}")

# start_time = time.perf_counter()
# num_digits = 20
# point_set = points_sorted(num_digits)
# print(
#     f"Number of digits : {num_digits}, number of points : {point_set.shape[0]}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Point Creation Time : {time_difference :0.6f}")

# start_time = time.perf_counter()
# num_digits = 29
# point_set = points_shuffled(num_digits)
# print(
#     f"Number of digits : {num_digits}, number of points : {point_set.shape[0]}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Point Creation Time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"first part : {first_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time: {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"first part : {first_part_version_2(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time: {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"first part : {first_part_version_2(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time: {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"second part : {second_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Second Part Time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"second part : {second_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Second Part Time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# # print(f"second part : {second_part_version_2(point_set)}")
# second_part_version_2(point_set)
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Second Part version 2 Time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# # print(f"second part : {second_part_version_2(point_set)}")
# second_part_version_2(point_set)
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Second Part version 2 Time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(
#     f"L2-star discrepancy : {1 / 3 ** 2 - first_part(point_set) + second_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time : {time_difference :0.6f}")


# print("******************************")
# start_time = time.perf_counter()
# num_digits = 10
# point_set = points(num_digits)
# print(
#     f"Number of digits : {num_digits}, number of points : {point_set.shape[0]}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"first part : {first_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time: {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"second part : {second_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Second Part Time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(
#     f"L2-star discrepancy : {1 / 3 ** 2 - first_part(point_set) + second_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time : {time_difference :0.6f}")


# print("******************************")
# start_time = time.perf_counter()
# num_digits = 20
# point_set = points(num_digits)
# print(
#     f"Number of digits : {num_digits}, number of points : {point_set.shape[0]}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Point construction time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"first part : {first_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time: {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(f"second part : {second_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Second Part Time : {time_difference :0.6f}")

# print()
# start_time = time.perf_counter()
# print(
#     f"L2-star discrepancy : {1 / 3 ** 2 - first_part(point_set) + second_part(point_set)}")
# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"First Part Time : {time_difference :0.6f}")
# print("******************************")
