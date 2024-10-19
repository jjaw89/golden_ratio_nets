
from cmath import sqrt
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

GOLDEN_RATIO = ((1+np.sqrt(5))/2).real


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

    return np.append(end_in_one, end_in_zero, axis=0)


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

    return np.append(end_in_one, end_in_zero, axis=0)


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
    return numbers.real[:, 0]


def points(num_digits):
    points = np.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_2d(num_digits))
    points = points[points[:, 0].argsort()]
    return points.real


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


def l2_star_discrepancy(point_set):
    return 1 / 3 ** 2 - first_part(point_set) + second_part(point_set)


def l2_star_discrepancy_version_2(point_set):
    return 1 / 3 ** 2 - first_part_version_2(point_set) + second_part_version_2(point_set)


# num_digits = 1
# point_set = points(num_digits)


vector_results = []
for num_digits in range(30):
    start_time = time.perf_counter()
    point_set = points(num_digits)
    vector_results += [[point_set.shape[0],
                        l2_star_discrepancy_version_2(point_set)]]
    mid_time = time.perf_counter() - start_time
    # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
    arr = np.array(vector_results)
    # print(arr)
    pd.DataFrame(arr).to_csv('results base phi.csv')
    print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")

# arr = np.array(vector_results)
# # print(arr)
# pd.DataFrame(arr).to_csv('results base phi.csv')


# start_time = time.perf_counter()
# num_digits = 20
# point_set = points(num_digits)
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
