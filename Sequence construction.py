from cmath import sqrt
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
    return numbers[:, 0]


def hammersley_points_sorted(num_digits):
    points = np.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_2d(num_digits))
    points = points[points[:, 0].argsort()]
    return points


def hammersley_points(num_digits):
    points = np.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_2d(num_digits))
    points = points[points[:, 0].argsort()]
    return points


def hammersley_points_shuffled(num_digits):
    points = np.matmul(no_consecutive_ones_b(
        num_digits), golden_multiplied_2d(num_digits))
    np.random.shuffle(points)
    return points


def plot_hammersley(num_digits, x_partition, ypartition):
    hammersley = hammersley_points(num_digits)
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


def convert_to_decimal(vectors):
    multiplied_matrix = np.array([[GOLDEN_RATIO ** (-k)]
                                  for k in range(1, vectors.shape[1]+1)])
    # print(multiplied_matrix)
    # print(np.matmul(vectors, multiplied_matrix))
    return np.matmul(vectors, multiplied_matrix)


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
    return a_i_plus


def plot_points(points, x_partition, ypartition):
    x = points[:, 0]
    y = points[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=7)
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def plot_points_2(points, num_digits, x_partition, ypartition):
    plt.figure(figsize=(6, 6))  # 9
    ax = plt.gca()
    ax.set_aspect(1)
    # print(fibonacci(num_digits+1))
    plt.plot(points[:fibonacci(num_digits+1), 0],
             points[:fibonacci(num_digits+1), 1], '.', markersize=15, c='b')
    plt.plot(points[fibonacci(num_digits+1):fibonacci(num_digits+2), 0],
             points[fibonacci(num_digits+1):fibonacci(num_digits+2), 1], '.', markersize=15, c='r')
    plt.grid()
    plt.xticks(ticks=partition(x_partition), labels=[])
    plt.yticks(ticks=partition(ypartition), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


def partition_numbers_2d(digits_to_partition, x_partition_digits, y_partition_digits):
    last_digit_x = no_consecutive_ones_reversed(x_partition_digits)[:, -1]
    last_digit_y = no_consecutive_ones_reversed(y_partition_digits)[:, -1]
    return np.array([[fibonacci(digits_to_partition+2-x_partition_digits-y_partition_digits-r-c) for r in last_digit_y] for c in last_digit_x])


def partition_numbers_1d(digits_to_partition, partition_digits):
    last_digit = no_consecutive_ones_reversed(partition_digits)[:, -1]
    return np.array([fibonacci(digits_to_partition+2-partition_digits-r) for r in last_digit])


def partition_digits(length):
    return no_consecutive_ones_reversed(length)


starting_points = np.array([
    [[0, 0], [0, 0]],
    [[1, 0], [1, 0]],
    [[0, 1], [0, 1]]])


for _ in range(17):
    start_time = time.perf_counter()
    # print(starting_points.shape[2])

    starting_x = starting_points[:, 0]
    starting_y = starting_points[:, 1]

    starting_x_zero = np.append(starting_x, np.zeros(
        (starting_x.shape[0], 1), dtype=int), axis=1)
    starting_y_zero = np.append(starting_y, np.zeros(
        (starting_y.shape[0], 1), dtype=int), axis=1)

    starting_points_zero = np.append(starting_x_zero, starting_y_zero, axis=1)
    starting_points_zero = np.array(
        [[starting_x_zero[n], starting_y_zero[n]] for n in range(starting_y_zero.shape[0])])
    final_num_digits = starting_x_zero.shape[1]
    # print(starting_points_zero)

    digits_to_add_x = no_consecutive_ones(final_num_digits)[
        fibonacci(final_num_digits+1):]
    digits_to_add_y = np.zeros_like(digits_to_add_x)
    digits_to_add_y[:, -1] += 1

    points_to_add = np.array([[digits_to_add_x[n], digits_to_add_y[n]]
                             for n in range(digits_to_add_y.shape[0])])
    # print(points_to_add)

    #################

    for m in range(1, final_num_digits-1):
        num_x_partition_digits = final_num_digits-1-m
        num_y_partition_digits = m
        time_1 = time.perf_counter()
        x_partition_digits = partition_digits(num_x_partition_digits)
        y_partition_digits = partition_digits(num_y_partition_digits)

        # print(x_partition_digits)
        # print(y_partition_digits)
        partition_count_matrix = partition_numbers_2d(
            final_num_digits, num_x_partition_digits, num_y_partition_digits)
        # print(partition_count_matrix)
        # plot_points_2(point_values,final_num_digits,num_x_partition_digits,num_y_partition_digits)
        for point_digits in starting_points_zero:
            for x_partition_index in range(fibonacci(num_x_partition_digits+2)):
                if np.array_equal(x_partition_digits[x_partition_index], point_digits[0, :num_x_partition_digits]):
                    break
            for y_partition_index in range(fibonacci(num_y_partition_digits+2)):
                if np.array_equal(y_partition_digits[y_partition_index], point_digits[1, :num_y_partition_digits]):
                    break
            partition_count_matrix[x_partition_index, y_partition_index] -= 1
        # print(partition_count_matrix)
        time_2 = time.perf_counter()
        print(
            f'({num_x_partition_digits},{num_y_partition_digits})-Time : {time_2-time_1:0.6f}')
        for add_point in points_to_add:
            # print()
            # print(f'starting point:{add_point}')
            for x_partition_index in range(fibonacci(num_x_partition_digits+2)):
                # print(x_partition_digits[x_partition_index])
                if np.array_equal(x_partition_digits[x_partition_index], add_point[0, :num_x_partition_digits]):
                    break
            # print(x_partition_index)
            for y_partition_index in range(fibonacci(num_y_partition_digits+2)):
                # print(y_partition_digits[y_partition_index])
                if np.array_equal(y_partition_digits[y_partition_index], add_point[1, :num_y_partition_digits]):
                    # print(y_partition_index)
                    break
        # print(partition_count_matrix[x_partition_index,y_partition_index])
            if partition_count_matrix[x_partition_index, y_partition_index] == 0:
                # print(x_partition_index)
                add_point[1, num_y_partition_digits-1] = 1

        # print(add_point)

    # points= np.append(starting_points_zero,points_to_add,axis=0)
    # x_values = convert_to_decimal(points[:,0])
    # y_values = convert_to_decimal(points[:,1])
    # point_values = np.append(x_values,y_values,axis=1)

    # plot_points_2(point_values,final_num_digits,num_x_partition_digits,num_y_partition_digits)
    #########################################

    # points= np.append(starting_points_zero,points_to_add,axis=0)
    # x_values = convert_to_decimal(points[:,0])
    # y_values = convert_to_decimal(points[:,1])
    # point_values = np.append(x_values,y_values,axis=1)

    # plot_points_2(point_values,final_num_digits,1,1)

    #########################################
    # print(starting_points_zero.shape)
    # print(points_to_add.shape)

    starting_points = np.append(starting_points_zero, points_to_add, axis=0)
    mid_time = time.perf_counter() - start_time
    print(f'{starting_points.shape}, time : {mid_time}')
points_up_to_now = starting_points

x_values = convert_to_decimal(points_up_to_now[:, 1])
# y_values = convert_to_decimal(points_up_to_now[:, 1])
# point_values = np.append(x_values, y_values, axis=1)

# print(point_values)

# starting_points = np.array([
#     [[0, 0], [0, 0]],
#     [[1, 0], [0, 1]],
#     [[0, 1], [1, 0]]])


# for _ in range(14):
#     start_time = time.perf_counter()
#     # print(starting_points.shape[2])

#     starting_x = starting_points[:, 0]
#     starting_y = starting_points[:, 1]

#     starting_x_zero = np.append(starting_x, np.zeros(
#         (starting_x.shape[0], 1), dtype=int), axis=1)
#     starting_y_zero = np.append(starting_y, np.zeros(
#         (starting_y.shape[0], 1), dtype=int), axis=1)

#     starting_points_zero = np.append(starting_x_zero, starting_y_zero, axis=1)
#     starting_points_zero = np.array(
#         [[starting_x_zero[n], starting_y_zero[n]] for n in range(starting_y_zero.shape[0])])
#     final_num_digits = starting_x_zero.shape[1]
#     # print(starting_points_zero)

#     digits_to_add_x = no_consecutive_ones(final_num_digits)[
#         fibonacci(final_num_digits+1):]
#     digits_to_add_y = np.zeros_like(digits_to_add_x)
#     digits_to_add_y[:, -1] += 1

#     points_to_add = np.array([[digits_to_add_x[n], digits_to_add_y[n]]
#                              for n in range(digits_to_add_y.shape[0])])
#     # print(points_to_add)

#     #################

#     for m in range(1, final_num_digits-1):
#         num_x_partition_digits = final_num_digits-1-m
#         num_y_partition_digits = m

#         x_partition_digits = partition_digits(num_x_partition_digits)
#         y_partition_digits = partition_digits(num_y_partition_digits)

#         # print(x_partition_digits)
#         # print(y_partition_digits)
#         partition_count_matrix = partition_numbers_2d(
#             final_num_digits, num_x_partition_digits, num_y_partition_digits)
#         # print(partition_count_matrix)
#         # plot_points_2(point_values,final_num_digits,num_x_partition_digits,num_y_partition_digits)
#         for point_digits in starting_points_zero:
#             for x_partition_index in range(fibonacci(num_x_partition_digits+2)):
#                 if np.array_equal(x_partition_digits[x_partition_index], point_digits[0, :num_x_partition_digits]):
#                     break
#             for y_partition_index in range(fibonacci(num_y_partition_digits+2)):
#                 if np.array_equal(y_partition_digits[y_partition_index], point_digits[1, :num_y_partition_digits]):
#                     break
#             partition_count_matrix[x_partition_index, y_partition_index] -= 1
#         # print(partition_count_matrix)
#         for add_point in points_to_add:
#             # print()
#             # print(f'starting point:{add_point}')
#             for x_partition_index in range(fibonacci(num_x_partition_digits+2)):
#                 # print(x_partition_digits[x_partition_index])
#                 if np.array_equal(x_partition_digits[x_partition_index], add_point[0, :num_x_partition_digits]):
#                     break
#             # print(x_partition_index)
#             for y_partition_index in range(fibonacci(num_y_partition_digits+2)):
#                 # print(y_partition_digits[y_partition_index])
#                 if np.array_equal(y_partition_digits[y_partition_index], add_point[1, :num_y_partition_digits]):
#                     # print(y_partition_index)
#                     break
#         # print(partition_count_matrix[x_partition_index,y_partition_index])
#             if partition_count_matrix[x_partition_index, y_partition_index] == 0:
#                 # print(x_partition_index)
#                 add_point[1, num_y_partition_digits-1] = 1

#         # print(add_point)

#     # points= np.append(starting_points_zero,points_to_add,axis=0)
#     # x_values = convert_to_decimal(points[:,0])
#     # y_values = convert_to_decimal(points[:,1])
#     # point_values = np.append(x_values,y_values,axis=1)

#     # plot_points_2(point_values,final_num_digits,num_x_partition_digits,num_y_partition_digits)
#     #########################################

#     # points= np.append(starting_points_zero,points_to_add,axis=0)
#     # x_values = convert_to_decimal(points[:,0])
#     # y_values = convert_to_decimal(points[:,1])
#     # point_values = np.append(x_values,y_values,axis=1)

#     # plot_points_2(point_values,final_num_digits,1,1)

#     #########################################
#     # print(starting_points_zero.shape)
#     # print(points_to_add.shape)

#     starting_points = np.append(starting_points_zero, points_to_add, axis=0)
#     mid_time = time.perf_counter() - start_time
#     print(f'{starting_points.shape}, time : {mid_time}')
# points_up_to_now = starting_points

# # x_values = convert_to_decimal(points_up_to_now[:, 0])
# y_values = convert_to_decimal(points_up_to_now[:, 1])

# point_values = np.append(x_values, y_values, axis=1)

# print(point_values)

vector_results = []
for num_digits in range(1, points_up_to_now.shape[2]+1):
    start_time = time.perf_counter()
    point_set = np.copy(point_values[:fibonacci(num_digits+2)])
    np.random.shuffle(point_set)
    # point_set = 1-point_set
    # point_set = np.flip(point_set, axis=0)
    vector_results += [[point_set.shape[0],
                        version_3_l2_star_discrepancy(point_set)]]
    mid_time = time.perf_counter() - start_time
    # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
    arr = np.array(vector_results)
    # print(arr)
    pd.DataFrame(arr).to_csv('L2 star discrepancy base phi Sequence.csv')
    print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
print(arr)

vector_results = []
for num_digits in range(1, points_up_to_now.shape[2]+1):
    start_time = time.perf_counter()
    point_set = 1 - np.copy(point_values[:fibonacci(num_digits+2)])
    np.random.shuffle(point_set)
    # point_set = 1-point_set
    # point_set = np.flip(point_set, axis=0)
    vector_results += [[point_set.shape[0],
                        version_3_l2_star_discrepancy(point_set)]]
    mid_time = time.perf_counter() - start_time
    # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
    arr = np.array(vector_results)
    # print(arr)
    pd.DataFrame(arr).to_csv('L2 star discrepancy base phi 1-Sequence.csv')
    print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
print(arr)

vector_results = []
for num_digits in range(1, points_up_to_now.shape[2]+1):
    start_time = time.perf_counter()
    point_set = 1 - np.copy(point_values[:fibonacci(num_digits+2)])
    np.random.shuffle(point_set)
    # point_set = 1-point_set
    # point_set = np.flip(point_set, axis=0)
    vector_results += [[point_set.shape[0],
                        version_3_l2_star_discrepancy(point_set)]]
    mid_time = time.perf_counter() - start_time
    # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
    arr = np.array(vector_results)
    # print(arr)
    pd.DataFrame(arr).to_csv('L2 star discrepancy base phi 1-Sequence.csv')
    print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
print(arr)

vector_results = []
for num_digits in range(1, points_up_to_now.shape[2]+1):
    start_time = time.perf_counter()
    point_set = np.copy(point_values[:fibonacci(num_digits+2)])
    point_set = point_set[point_set[:, 0].argsort()]
    vector_results += [[point_set.shape[0],
                        star_discrepancy(point_set)]]
    mid_time = time.perf_counter() - start_time
    # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
    arr = np.array(vector_results)
    # print(arr)
    pd.DataFrame(arr).to_csv('Discrepancy Sequence.csv')
    print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
print(arr)

vector_results = []
for num_digits in range(1, points_up_to_now.shape[2]+1):
    start_time = time.perf_counter()
    point_set = 1-np.copy(point_values[:fibonacci(num_digits+2)])
    point_set = point_set[point_set[:, 0].argsort()]
    vector_results += [[point_set.shape[0],
                        star_discrepancy(point_set)]]
    mid_time = time.perf_counter() - start_time
    # [num_digits, point_set.size[0], l2_star_discrepancy(point_set)]
    arr = np.array(vector_results)
    # print(arr)
    pd.DataFrame(arr).to_csv('Discrepancy 1-Sequence.csv')
    print(f"num digits : {num_digits}, time so far : {mid_time:0.6f}")
print(arr)
