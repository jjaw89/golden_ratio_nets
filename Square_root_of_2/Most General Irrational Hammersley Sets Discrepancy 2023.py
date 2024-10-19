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


def plot_points(points, a, b, x_partition, ypartition, size):
    x = points[:, 0]
    y = points[:, 1]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect(1)
    plt.plot(x, y, '.', markersize=size)
    plt.grid()
    plt.xticks(ticks=increasing_whole_numbers_values_2(
        x_partition, a, b), labels=[])
    plt.yticks(ticks=increasing_whole_numbers_values_2(
        ypartition, a, b), labels=[])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.show()


@njit()
def num_whole_numbers(num_digits, a, b):

    if num_digits == 0:
        return 1
    elif num_digits == 1:
        return a+1
    else:
        p_minus_2 = 1
        p_minus_1 = a+1
        for _ in range(num_digits-1):
            p = a*p_minus_1 + b*p_minus_2
            p_minus_2 = p_minus_1
            p_minus_1 = p

        return p


@njit()
def num_whole_numbers(num_digits, a, b):

    if num_digits == 0:
        return 1
    elif num_digits == 1:
        return a+1
    else:
        p_minus_2 = 1
        p_minus_1 = a+1
        for _ in range(num_digits-1):
            p = a*p_minus_1 + b*p_minus_2
            p_minus_2 = p_minus_1
            p_minus_1 = p

        return p


@njit()
def num_whole_numbers_2(num_digits, a, b):

    if num_digits == 0:
        return 1
    elif num_digits == 1:
        return a+1
    else:
        # one digit
        highest_digit_less_than_b = b
        num_numbers = a+1
        for _ in range(num_digits-1):
            next_num_numbers = a * num_numbers + highest_digit_less_than_b
            highest_digit_less_than_b = b * num_numbers
            num_numbers = next_num_numbers

        return num_numbers


@njit()
def num_whole_numbers_2_highest_digit_less_than_b(num_digits, a, b):
    return num_whole_numbers_2(num_digits - 1, a, b) * b


@njit()
def num_whole_numbers_2_not_ending_in_a(num_digits, a, b):
    return num_whole_numbers_2(num_digits, a, b) - num_whole_numbers_2_highest_digit_less_than_b(num_digits-1, a, b)


@njit()
def num_whole_numbers(num_digits, a, b):

    if num_digits == 0:
        return 1
    elif num_digits == 1:
        return a+1
    else:
        p_minus_2 = 1
        p_minus_1 = a+1
        for _ in range(num_digits-1):
            p = a*p_minus_1 + b*p_minus_2
            p_minus_2 = p_minus_1
            p_minus_1 = p

        return p


@njit()
def num_whole_numbers_ending_less_than_b(num_digits, a, b):

    if num_digits == 0:
        return 1
    elif num_digits == 1:
        return b
    else:
        p_minus_2 = 1
        p_minus_1 = b
        for _ in range(num_digits-1):
            p = a*p_minus_1 + b*p_minus_2
            p_minus_2 = p_minus_1
            p_minus_1 = p
        p = b*p_minus_1

        return p


@njit()
def van_der_corput_ending_less_than_a(num_digits, a, b):

    if num_digits == 0:
        return 1
    elif num_digits == 1:
        return a
    else:
        total_minus_1 = a+1
        p_minus_1 = a
        for _ in range(num_digits-1):
            p = b*total_minus_1 + (a-b)*p_minus_1
            total_minus_1 = b*total_minus_1 + (a-b+1)*p_minus_1
            p_minus_1 = p

        return p


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


@njit()  # Resulting decimals increasing
def increasing_whole_numbers(num_digits, a, b):
    if num_digits == 0:
        return np.array([0], dtype='int64')

    numbers = np.zeros(num_whole_numbers(
        num_digits, a, b), dtype='int64')
    base = a+1
    numbers[:a+1] += np.arange(0, a+1, 1)
    # m is current number of digits
    for m in range(1, num_digits):
        length_1 = num_whole_numbers(m, a, b)
        length_2 = num_whole_numbers_ending_less_than_b(m, a, b)
        length_2 = van_der_corput_ending_less_than_a(m, a, b)
        # print(length_1)
        # print(length_2)
        for j in range(1, a):
            numbers[j*length_1: (j+1) *
                    length_1] += numbers[: length_1] + j*base**m
        numbers[a*length_1: a*length_1 +
                length_2] += numbers[: length_2] + a*base**m
        # print(numbers)
    return numbers


@njit()  # Resulting decimals increasing
def increasing_whole_numbers_2(num_digits, a, b):
    if num_digits == 0:
        return np.array([0], dtype='int64')

    numbers = np.zeros(num_whole_numbers_2(
        num_digits, a, b), dtype='int64')
    base = a+1
    numbers[:a+1] += np.arange(0, a+1, 1)
    # m is current number of digits
    for m in range(1, num_digits):
        length_1 = num_whole_numbers_2(m, a, b)
        length_2 = num_whole_numbers_2_highest_digit_less_than_b(m, a, b)
        # print(length_1)
        # print(length_2)
        for j in range(1, a):
            numbers[j*length_1: (j+1) *
                    length_1] += numbers[: length_1] + j*base**m
        numbers[a*length_1: a*length_1 +
                length_2] += numbers[: length_2] + a*base**m
        # print(numbers)
    return numbers


@njit()  # Resulting decimals increasing
def increasing_whole_numbers_values_2(num_digits, a, b):
    if num_digits == 0:
        return np.array([0], dtype='float')

    numbers = np.zeros(num_whole_numbers_2(
        num_digits, a, b), dtype='float')
    base = ((math.sqrt(a ** 2+4*b)+a) / 2).real
    numbers[:a+1] += np.arange(0, a+1, 1) * base ** (-num_digits)
    # m is current number of digits
    for m in range(1, num_digits):
        length_1 = num_whole_numbers_2(m, a, b)
        length_2 = num_whole_numbers_2_highest_digit_less_than_b(m, a, b)
        # print(length_1)
        # print(length_2)
        for j in range(1, a):
            numbers[j*length_1: (j+1) *
                    length_1] += numbers[: length_1] + j*base**(-num_digits+m)
        numbers[a*length_1: a*length_1 +
                length_2] += numbers[: length_2] + a*base**(-num_digits+m)
        # print(numbers)
    return numbers


@njit()  # Resulting decimals increasing
def increasing_whole_numbers_values(num_digits, a, b):
    if num_digits == 0:
        return np.array([0], dtype='float')

    numbers = np.zeros(num_whole_numbers(
        num_digits, a, b), dtype='float')
    base = ((math.sqrt(a ** 2+4*b)+a) / 2).real
    numbers[:a+1] += np.arange(0, a+1, 1) * base ** (-num_digits)
    # m is current number of digits
    for m in range(1, num_digits):
        length_1 = num_whole_numbers(m, a, b)
        length_2 = num_whole_numbers_ending_less_than_b(m, a, b)
        length_2 = van_der_corput_ending_less_than_a(m, a, b)
        # print(length_1)
        # print(length_2)
        for j in range(1, a):
            numbers[j*length_1: (j+1) *
                    length_1] += numbers[: length_1] + j*base**(-num_digits + m)
        numbers[a*length_1: a*length_1 +
                length_2] += numbers[: length_2] + a*base**(-num_digits + m)
        # print(numbers)

    return numbers


@njit()  # Must be flipped
def van_der_corput_numbers(num_digits, a, b):
    if num_digits == 0:
        return np.array([0], dtype='int64')

    numbers = np.zeros(num_whole_numbers(
        num_digits, a, b), dtype='int64')
    base = a+1
    numbers[:a+1] += np.arange(0, a+1, 1)
    # m is current number of digits
    for m in range(1, num_digits):
        length_1 = num_whole_numbers(m, a, b)
        length_2 = van_der_corput_ending_less_than_a(m, a, b)
        # print(length_1)
        # print(length_2)
        for j in range(1, b):
            numbers[j*length_1: (j+1) *
                    length_1] += numbers[: length_1] + (j)*base**m
        for j in range(0, a-b+1):
            numbers[b*length_1+j*length_2: b*length_1 +
                    (j+1)*length_2] += numbers[: length_2] + (j+b)*base**m
        # print(numbers)

    return numbers


@njit()  # Must be flipped
def van_der_corput_values(num_digits, a, b):
    base = ((math.sqrt(a ** 2+4*b)+a) / 2).real
    if num_digits == 0:
        return np.array([0], dtype='float')

    values = np.zeros(num_whole_numbers(
        num_digits, a, b), dtype='float')
    values[:a+1] += np.arange(0, a+1, 1)*base ** (-1)
    # m is current number of digits
    for m in range(1, num_digits):
        length_1 = num_whole_numbers(m, a, b)
        length_2 = van_der_corput_ending_less_than_a(m, a, b)
        # print(length_1)
        # print(length_2)
        for j in range(1, b):
            values[j*length_1: (j+1) *
                   length_1] += values[: length_1] + (j)*base**(-m-1)
        for j in range(0, a-b+1):
            values[b*length_1+j*length_2: b*length_1 +
                   (j+1)*length_2] += values[: length_2] + (j+b)*base**(-m-1)
        # print(values)

    return values


@njit(parallel=True)
def numbers_to_digits(numbers, num_digits, a):
    digits = np.zeros((numbers.shape[0], num_digits), dtype='int64')
    for j in prange(numbers.shape[0]):
        number = numbers[j]
        for m in range(num_digits):
            digits[j, num_digits-1 - m] = number % (a+1)
            number //= (a+1)
    return digits


@njit()
def hammersley_values(num_digits, a, b):
    x_values = increasing_whole_numbers_values(num_digits, a, b)
    y_values = van_der_corput_values(num_digits, a, b)
    points = np.array([[x_values[n], y_values[n]]
                      for n in range(x_values.shape[0])], dtype='float')
    return points


@njit()
def hammersley_values_2(num_digits, a, b):
    x_values = increasing_whole_numbers_values_2(num_digits, a, b)
    y_values = van_der_corput_values(num_digits, a, b)
    points = np.array([[x_values[n], y_values[n]]
                      for n in range(x_values.shape[0])], dtype='float')
    return points


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


def scramble_numbers_m_digits_cp(numbers_gpu, m, a, b):

    random_zeros_and_ones_gpu = cp.random.randint(
        0, 2, (FIBONACCI[m],), dtype='uint32')
    random_zeros_and_ones_gpu *= 2**(m-1)
    permutation_gpu = cp.copy(
        ALL_THIRTY_TWO_DIGIT_NUMBERS_GPU[:FIBONACCI[m+3]])
    permutation_gpu[:FIBONACCI[m]] += random_zeros_and_ones_gpu
    permutation_gpu[FIBONACCI[m+1]:FIBONACCI[m+2]] -= random_zeros_and_ones_gpu
    numbers_gpu = permutation_gpu[cp.searchsorted(
        ALL_THIRTY_TWO_DIGIT_NUMBERS_GPU[:FIBONACCI[m+3]], numbers_gpu % 2**(m+1))] + (numbers_gpu // 2**(m+1))*2**(m+1)
    return numbers_gpu


def scramble_mth_digit(numbers, m, a, b):

    numbers_gpu = cp.asarray(numbers)
    random_0_to_a = cp.array([cp.random.permutation(a)
                             for _ in range(num_whole_numbers_2_not_ending_in_a(m-1, a, b))], dtype='uint8')
    array_a = cp.full((num_whole_numbers_2_not_ending_in_a(
        m-1, a, b), 1), a,  dtype='uint8')
    random_0_to_a = cp.append(random_0_to_a, array_a, axis=1)
    print(random_0_to_a)
    random_0_to_a = cp.transpose(random_0_to_a).reshape(
        np.product(random_0_to_a.shape))
    print(random_0_to_a)
    random_0_to_b = cp.array([cp.random.permutation(b)
                             for _ in range(num_whole_numbers_2_highest_digit_less_than_b(m-2, a, b))], dtype='uint8')
    random_0_to_a = cp.transpose(random_0_to_b).reshape(
        np.product(random_0_to_b.shape))

    permutation_gpu = cp.zeros(numbers_gpu.shape[0], dtype='int32')


def scramble_non_zero_digits_no_back_tracking_cp(numbers, num_digits, a, b):
    n = num_digits
    numbers = np.array(numbers, dtype='uint32')
    numbers_gpu = cp.asarray(numbers[:, 0])
    for m in range(1, n+1):
        numbers_gpu = scramble_numbers_m_digits_cp(
            m, numbers_gpu)
    numbers[:, 0] = cp.asnumpy(numbers_gpu)


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

# num_digits = 3
# a = 3
# b = 2
# # print(num_whole_numbers(num_digits, a, b))
# # # print(num_whole_numbers_ending_less_than_b(num_digits, a, b))
# # print(numbers_to_digits(increasing_whole_numbers(num_digits, a, b), num_digits, a))
# print(np.flip(numbers_to_digits(van_der_corput_numbers(
#     num_digits, a, b), num_digits, a), axis=1))
# print()
# # print(numbers_to_digits(van_der_corput_numbers(
# #     num_digits, a, b), num_digits, a))
# print()
# print(van_der_corput_numbers(
#     num_digits, a, b))
# print()
# print(numbers_to_digits(increasing_whole_numbers(num_digits, a, b), num_digits, a))
# print(increasing_whole_numbers_2(num_digits, a, b)[-20:])
# # print(increasing_whole_numbers(num_digits, a, b))
# print()
# print(numbers_to_digits(increasing_whole_numbers_2(num_digits, a, b), num_digits, a))
# # print(van_der_corput_ending_less_than_a(3, a, b))
# # print(num)

# for num_digits in range(4):
#     print(num_whole_numbers_ending_less_than_b(num_digits, a, b))
# for num_digits in range(4):
#     print(van_der_corput_ending_less_than_a(num_digits, a, b))
# # print(numbers_to_digits(van_der_corput_numbers(
# #     num_digits, a, b), num_digits, a))
# # print(van_der_corput_values(num_digits, a, b))
# # print(increasing_whole_numbers_values(num_digits, a, b))

# num_digits = 4
# a = 2
# b = 2
# plot_points(hammersley_values(num_digits, a, b), a, b, 1, 1, 10)

# a = 5
# b = 5
# print(num_whole_numbers(7, a, b))


# a = 3
# b = 1
# for n in range(15):
#     if num_whole_numbers_2(n, a, b) >= 700000:
#         break
# for num_digits in range(1, n-1):

#     points = hammersley_values_2(num_digits, a, b)

#     start_time = time.perf_counter()
#     result = star_discrepancy_largest_rest(points)
#     total_time_2 = time.perf_counter() - start_time

#     if result[0, 0] > result[1, 0]:
#         print(
#             f'{total_time_2},{a},{b}, {points.shape[0]}, {num_digits}, c, {result[0,0]}, {result[0,1]} , {result[0,2]}')
#     else:
#         print(
#             f'{total_time_2},{a},{b},{points.shape[0]}, {num_digits}, o, {result[0,0]}, {result[0,1]} , {result[0,2]}')
# print()

# plot_points(hammersley_values(4, a, b), a, b, 0, 0, 10)
# plot_points(hammersley_values_2(10, 1, 1), a, b, 2, 0, 10)
# plot_points(hammersley_values_2(6, 2, 1), a, b, 2, 0, 10)
# plot_points(hammersley_values_2(6, 2, 2), a, b, 2, 0, 10)
# plot_points(hammersley_values_2(5, 3, 1), a, b, 2, 0, 10)
# plot_points(hammersley_values_2(5, 3, 2), a, b, 2, 0, 10)
# plot_points(hammersley_values_2(5, 3, 3), a, b, 2, 0, 10)
# # plot_points(hammersley_values_2(4, a, b), a, b, 2, 0, 10)
# # plot_points(hammersley_values_2(4, a, b), a, b, 3, 0, 10)
# perm = np.array([np.random.permutation(3)+1
#                 for _ in range(3)]) * (10 ** np.arange(3).reshape((3, 1)))
# print(perm)
# print(np.transpose(perm).reshape(9))

# for num_digits in range(10):
#     print(f'{num_whole_numbers(num_digits,a,b)} : {num_whole_numbers_2(num_digits,a,b)}')
# for num_digits in range(10):
#     print(f'{num_whole_numbers_ending_less_than_b(num_digits, a, b)} : {num_whole_numbers_2_highest_digit_less_than_b(num_digits, a, b)}')
# for num_digits in range(10):
#     print(van_der_corput_ending_less_than_a

# num_digits = 3
# a = 2
# b = 2
# print()
# print(numbers_to_digits(van_der_corput_numbers(
#     num_digits, a, b), num_digits, a))
# print(numbers_to_digits(increasing_whole_numbers_2(num_digits, a, b), num_digits, a))


a = 2
b = 2
# for n in range(15):
#     if num_whole_numbers_2(n, a, b) >= 2000000:
#         break
# for num_digits in range(1, n-1):
#     num_digits = 13
#     points = hammersley_values_2(num_digits, a, b)

#     start_time = time.perf_counter()
#     result = star_discrepancy_largest_rest(points)
#     total_time_2 = time.perf_counter() - start_time

#     if result[0, 0] > result[1, 0]:
#         print(
#             f'{total_time_2},{a},{b}, {points.shape[0]}, {num_digits}, c, {result[0,0]}, {result[0,1]} , {result[0,2]}')
#     else:
#         print(
#             f'{total_time_2},{a},{b},{points.shape[0]}, {num_digits}, o, {result[0,0]}, {result[0,1]} , {result[0,2]}')
# print()

for num_digits in range(1, 14):
    points = hammersley_values_2(num_digits, a, b)

    start_time = time.perf_counter()
    result = l2_star_discrepancy_cuda(points)
    total_time_2 = time.perf_counter() - start_time
    print(f'{result},{total_time_2},{num_digits}')


# for m in range(3):
#     start_time = time.perf_counter()
#     l2 = l2_star_discrepancy_cuda(points[:FIBONACCI[m+2]])
#     star = star_discrepancy_largest_rest(points[:FIBONACCI[m+2]])[0, 0]
#     end_time = time.perf_counter()
#     print(f'{start_time-end_time}, {m}, {star}, {l2}')
