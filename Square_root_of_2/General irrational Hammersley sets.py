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
plt.style.use('seaborn-whitegrid')


def plot_points(points, largest_n, x_partition, ypartition, size):
    x = points[:, 0]
    y = points[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=size)
    plt.grid()
    plt.xticks(ticks=partition(x_partition, largest_n), labels=[])
    plt.yticks(ticks=partition(ypartition, largest_n), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


@njit()
def num_valid_sequences(num_digits, largest_n):
    zero_digit = 1
    one_digit = largest_n+1

    if num_digits == 0:
        return zero_digit
    elif num_digits == 1:
        return one_digit
    else:
        p_minus_2 = zero_digit
        p_minus_1 = one_digit
        for _ in range(num_digits-1):
            p = largest_n*p_minus_1 + p_minus_2
            p_minus_2 = p_minus_1
            p_minus_1 = p

        return p


@njit()
def num_valid_sequences_2(num_digits, largest_n):
    zero_digit = 1
    one_digit = largest_n+1

    if num_digits == 0:
        return zero_digit
    elif num_digits == 1:
        return one_digit
    else:
        p_minus_2 = zero_digit
        p_minus_1 = one_digit
        for _ in range(num_digits-1):
            p = largest_n*p_minus_1 + p_minus_2
            p_minus_2 = p_minus_1
            p_minus_1 = p

        return p


# def sequences_n_must_be_preceded_by_zero_2(num_digits, largest_n):
#     if num_digits == 0:
#         return np.array([0], dtype='int64')
#     end_in_0 = np.array([0], dtype='int64')
#     end_in_n = np.arange(1, largest_n+1, 1)

#     digits_to_add = np.arange(
#         1, largest_n, 1).reshape((largest_n-1, 1))
#     base = largest_n+1

#     for m in range(1, num_digits):
#         # next digit is 0
#         next_end_in_0 = np.append(end_in_0, end_in_n, axis=0)

#         # next digit is 1-(largest_n-1)
#         all_vectors = np.append(end_in_0, end_in_n, axis=0)
#         all_vectors_extended = np.array(
#             [all_vectors]*(largest_n-1), dtype='int64')+digits_to_add * (base) ** m

#         all_vectors_extended = all_vectors_extended.reshape(
#             np.prod(all_vectors_extended.shape))
#         # next digit is largest_n
#         end_in_largest_n = end_in_0 + largest_n * base**m

#         end_in_n = np.append(all_vectors_extended, end_in_largest_n, axis=0)
#         end_in_0 = next_end_in_0

#     final_numbers = np.append(end_in_0, end_in_n, axis=0)
#     return final_numbers


@njit()  # Resulting decimals increasing
def sequences_largest_n_must_be_preceded_by_zero(num_digits, largest_n):
    if num_digits == 0:
        return np.array([0], dtype='int64')

    numbers = np.zeros(num_valid_sequences(
        num_digits, largest_n), dtype='int64')
    base = largest_n+1
    numbers[:num_valid_sequences(
        1, largest_n)] += np.arange(0, num_valid_sequences(1, largest_n), 1)
    for m in range(1, num_digits+1):
        length_1 = num_valid_sequences(m, largest_n)
        length_2 = num_valid_sequences(m-1, largest_n)
        for j in range(1, largest_n):
            numbers[j*length_1: (j+1) *
                    length_1] += numbers[: length_1] + j*base**m
        numbers[largest_n*length_1: largest_n*length_1 +
                length_2] += numbers[: length_2] + largest_n*base**m

    return numbers


@njit()
def reversed_sequences_counting(num_digits, largest_n):
    end_in_largest = 1
    end_in_other = largest_n
    for _ in range(num_digits-1):
        # extend with a zero
        end_in_other_next = end_in_largest + end_in_other
        # extend with not zero and not largest
        end_in_other_next += (largest_n-1)*end_in_other
        # extend with largest
        end_in_largest = end_in_other
        end_in_other = end_in_other_next
    return np.array([end_in_other, end_in_largest], dtype='int64')


@njit()
def sequences_largest_n_must_be_preceded_by_zero_lexsort(num_digits, largest_n):
    if num_digits == 0:
        return np.array([0], dtype='int64')

    numbers = np.zeros(num_valid_sequences(
        num_digits, largest_n), dtype='int64')
    base = largest_n+1
    numbers[:num_valid_sequences(
        1, largest_n)] += np.arange(0, num_valid_sequences(1, largest_n), 1) * base ** (num_digits-1)
    # print(numbers)
    for m in range(1, num_digits+1):
        length_1 = num_valid_sequences(m, largest_n)
        length_2 = num_valid_sequences(m+1, largest_n)
        counting_1 = reversed_sequences_counting(m, largest_n)
        counting_2 = reversed_sequences_counting(m+1, largest_n)
        for j in range(largest_n-1):
            numbers[length_1 + j*counting_1[0]: length_1+(j+1) *
                    counting_1[0]] += numbers[: counting_1[0]] + (j+1)*base**(num_digits - m-1)
        numbers[length_2-counting_2[1]:length_2] += numbers[: counting_2[1]
                                                            ] + largest_n*base**(num_digits - m-1)
        # print(numbers)

    return numbers


@njit(parallel=True)
def numbers_to_digits(numbers, largest_n, num_digits):
    digits = np.zeros((numbers.shape[0], num_digits), dtype='int64')
    for j in prange(numbers.shape[0]):
        number = numbers[j]
        for m in range(num_digits):
            digits[j, num_digits-1 - m] = number % (largest_n+1)
            number //= (largest_n+1)
    return digits


@njit(parallel=True)
def numbers_to_decimals(numbers, largest_n, num_digits):
    base = ((largest_n+math.sqrt(largest_n**2 + 4)) / 2).real
    decimals = np.zeros_like(numbers, dtype='float')
    for m in prange(numbers.shape[0]):
        number = numbers[m]
        for j in range(num_digits):
            decimals[m] += (number % (largest_n + 1)) * base ** (-num_digits+j)
            number //= (largest_n + 1)
    return decimals


@njit()
def hammersley_numbers(num_digits, largest_n):
    x_numbers = sequences_largest_n_must_be_preceded_by_zero(
        num_digits, largest_n)
    y_numbers = sequences_largest_n_must_be_preceded_by_zero_lexsort(
        num_digits, largest_n)
    point_numbers = np.array([[x_numbers[n], y_numbers[n]]
                             for n in range(x_numbers.shape[0])], dtype='int64')
    return point_numbers


@njit()
def hammersley_points(num_digits, largest_n):
    point_numbers = hammersley_numbers(num_digits, largest_n)
    point_decimals = np.zeros_like(point_numbers, dtype='float')
    point_decimals[:, 0] += numbers_to_decimals(
        point_numbers[:, 0], largest_n, num_digits)
    point_decimals[:, 1] += numbers_to_decimals(
        point_numbers[:, 1], largest_n, num_digits)
    return point_decimals


@njit()
def hammersley_digits(num_digits, largest_n):
    point_numbers = hammersley_numbers(num_digits, largest_n)
    # print(point_numbers)
    x_digits = numbers_to_digits(point_numbers[:, 0], largest_n, num_digits)
    y_digits = numbers_to_digits(point_numbers[:, 1], largest_n, num_digits)
    x_digits = x_digits.reshape((x_digits.shape[0], 1, num_digits))
    y_digits = y_digits.reshape((x_digits.shape[0], 1, num_digits))
    digits = np.append(x_digits, y_digits, axis=1)
    return digits


@njit()
def partition(num_digits, largest_n):
    arr = numbers_to_decimals(sequences_largest_n_must_be_preceded_by_zero(
        num_digits, largest_n), largest_n, num_digits)
    one = np.ones(1, dtype='float')
    return np.append(arr, one, axis=0)


@njit()
def star_discrepancy_largest_rest(point_set):
    n = point_set.shape[0]

    # points = point_set[point_set[:, 0].argsort()]
    points = point_set

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


num_digits = 3
largest_n = 2

print(hammersley_digits(num_digits, largest_n)[:, 1])
# print(numbers_to_digits(sequences_largest_n_must_be_preceded_by_zero(
#     num_digits, largest_n), largest_n, num_digits))

# print(hammersley_digits(4, 2))


######################################################

# largest_n = 10
# for num_digits in range(1, 6):
#     print(f'{num_digits}')
#     plot_points(hammersley_points(num_digits, largest_n), largest_n, 0, 0, 7)

# largest_n = 3
# num_digits = 3
# for j in range(num_digits):
#     print(f'{j} : {num_digits - 1 - j}')
#     plot_points(hammersley_points(num_digits, largest_n),
#                 largest_n, j, num_digits-1-j, 7)

########################################################

# for num_digits in range(17, 19):

#     points = hammersley_points(num_digits, 2)
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
#     result = star_discrepancy_largest_rest(points)
#     total_time_2 = time.perf_counter() - start_time

#     # print(num_digits)
#     if result[0, 0] > result[1, 0]:
#         print(
#             f'{total_time_2}, {points.shape[0]}, {num_digits}, c, {result[0,0]}, {result[0,1]} , {result[0,2]}')
#     else:
#         print(
#             f'{total_time_2},{points.shape[0]}, {num_digits}, o, {result[0,0]}, {result[0,1]} , {result[0,2]}')
#     # print(f'{num_digits} : {total_time_1}')
#     # print(f'{num_digits} :  {total_time_2}')
#     # print(result)
#     # print()

# # for num_digits in range(12, 13):
# #     points = hammersley_points(num_digits, 3)

# #     # start_time = time.perf_counter()
# #     result = version_3_l2_star_discrepancy(points)
# #     # total_time_2 = time.perf_counter() - start_time

# #     print(
# #         f'{points.shape[0]}, {result}')
