import qmcpy
from cmath import sqrt
from dataclasses import replace
from msilib.schema import Binary
# from selectors import EpollSelector
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
    numbers[:a+1] += np.arange(0, a+1, 1) * base ** (num_digits-1)
    # m is current number of digits
    for m in range(1, num_digits):
        length_1 = num_whole_numbers_2(m, a, b)
        length_2 = num_whole_numbers_2_highest_digit_less_than_b(m, a, b)
        # print(length_1)
        # print(length_2)
        for j in range(1, a):
            numbers[j*length_1: (j+1) *
                    length_1] += numbers[: length_1] + j*base**(num_digits-1-m)
        numbers[a*length_1: a*length_1 +
                length_2] += numbers[: length_2] + a*base**(num_digits-1-m)
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
def hammersley_values_rationals(num_digits, a, b):

    y_values = van_der_corput_values(num_digits, a, b)
    x_values = np.array([k/y_values.shape[0]
                        for k in range(y_values.shape[0])], dtype='float')
    points = np.array([[x_values[n], y_values[n]]
                      for n in range(x_values.shape[0])], dtype='float')
    return points


@njit()
def hammersley_values_rationals_2(num_digits, a, b):
    y_values = van_der_corput_values(num_digits, a, b)
    y_values_order = np.argsort(y_values)
    x_values = np.array([k/y_values.shape[0]
                        for k in range(y_values.shape[0])], dtype='float')
    y_values = x_values[y_values_order]
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
def hammersley_numbers(num_digits, a, b):
    x_values = increasing_whole_numbers_2(num_digits, a, b)
    y_values = van_der_corput_numbers(num_digits, a, b)
    points = np.array([[x_values[n], y_values[n]]
                      for n in range(x_values.shape[0])], dtype='int64')
    return points


@njit()
def points_to_values(point_numbers, num_digits, a, b):
    point_numbers_2 = np.copy(point_numbers)
    point_values = np.zeros_like(point_numbers, dtype='float')
    base = ((math.sqrt(a ** 2+4*b)+a) / 2).real
    for m in range(point_numbers.shape[0]):
        numbers = point_numbers_2[m]
        for j in range(num_digits+1):
            point_values[m] += (numbers % (a+1)) * base ** (-j-1)
            numbers //= (a+1)
    return point_values


# @njit()
def hammersley_digits(num_digits, a, b):
    point_numbers = hammersley_numbers(num_digits, a, b)
    x_digits = numbers_to_digits(point_numbers[:, 0], num_digits, a)
    x_digits = x_digits.reshape((x_digits.shape[0], 1, x_digits.shape[1]))
    x_digits = np.flip(x_digits, axis=2)
    y_digits = numbers_to_digits(point_numbers[:, 1], num_digits, a)
    y_digits = y_digits.reshape((y_digits.shape[0], 1, y_digits.shape[1]))
    y_digits = np.flip(y_digits, axis=2)
    return np.append(x_digits, y_digits, axis=1)


@njit()
def star_discrepancy_largest_rest(point_set):
    n = point_set.shape[0]

    points = point_set[point_set[:, 0].argsort()]
    # points = point_set

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


def scramble_mth_digit_general(numbers, m, a, b):

    numbers_gpu = cp.asarray(numbers)

    length_1 = van_der_corput_ending_less_than_a(m-1, a, b)
    length_2 = num_whole_numbers(m-1, a, b) - \
        van_der_corput_ending_less_than_a(m-1, a, b)
    # print(f'length_1={length_1} : length_2={length_2}')

    random_0_to_a = cp.array([cp.random.permutation(a)
                             for _ in range(length_1)], dtype='uint32')
    array_a = cp.full((length_1, 1), a,  dtype='uint32')
    random_0_to_a = cp.append(random_0_to_a, array_a, axis=1)
    # print('random_0_to_a')
    # print(random_0_to_a)
    random_0_to_a = cp.transpose(random_0_to_a).reshape(
        np.product(random_0_to_a.shape))
    # print(random_0_to_a)

    random_0_to_b = cp.array([cp.random.permutation(b)
                             for _ in range(length_2)], dtype='uint32')
    # print('random_0_b')
    # print(random_0_to_b)
    random_0_to_b = cp.transpose(random_0_to_b).reshape(
        np.product(random_0_to_b.shape))
    # print(random_0_to_b)

    permutation_gpu = cp.zeros(num_whole_numbers(m, a, b), dtype='int32')
    van_der_corput = cp.asarray(van_der_corput_numbers(m, a, b))
    multiplier = (a+1)**(m-1)
    length_1 = num_whole_numbers(m-1, a, b)
    length_2 = van_der_corput_ending_less_than_a(m-1, a, b)
    length_3 = length_1-length_2
    # print(f'length_1={length_1} : length_2={length_2} : length_3={length_3}')

    # print(van_der_corput_one_less)
    # previous digit is not a
    for j in range(b):
        # print(random_0_to_a[j*length_2:(j+1)*length_2])
        permutation_gpu[j*length_1: j*length_1 +
                        length_2] += random_0_to_a[j*length_2:(j+1)*length_2]*multiplier + van_der_corput[:length_2]
    for j in range(a-b+1):
        permutation_gpu[b*length_1+j*length_2:b*length_1 +
                        (j+1)*length_2] += random_0_to_a[b*length_2 + j*length_2: b*length_2 + (j+1)*length_2]*multiplier + van_der_corput[:length_2]
    # previous digit is a
    for j in range(b):
        permutation_gpu[j*length_1 + length_2:(
            j+1)*length_1] += van_der_corput[length_2:length_1] + random_0_to_b[j*length_3:(j+1)*length_3]*multiplier

    # for j in range(b):
    #     # print(random_0_to_a[j*length_2:(j+1)*length_2])
    #     permutation_gpu[j*length_1: j*length_1 +
    #                     length_2] += random_0_to_a[j*length_2:(j+1)*length_2]*multiplier + ALL_VAN_DER_CORPUT_GPU[:length_2]
    # for j in range(a-b+1):
    #     permutation_gpu[b*length_1+j*length_2:b*length_1 +
    #                     (j+1)*length_2] += random_0_to_a[b*length_2 + j*length_2: b*length_2 + (j+1)*length_2]*multiplier + ALL_VAN_DER_CORPUT_GPU[:length_2]
    # # previous digit is a
    # for j in range(b):
    #     permutation_gpu[j*length_1 + length_2:(
    #         j+1)*length_1] += ALL_VAN_DER_CORPUT_GPU[length_2:length_1] + random_0_to_b[j*length_3:(j+1)*length_3]*multiplier

    # print(permutation_gpu)
    # print(van_der_corput_numbers(m, a, b))
    # print(numbers_to_digits(cp.asnumpy(permutation_gpu), m, a))
    # van_der_corput_all = cp.asarray(van_der_corput_numbers(m, a, b))

    # print(f'{permutation_gpu % (a+1)**(m)}')
    # print(van_der_corput_all)
    numbers_gpu = permutation_gpu[cp.searchsorted(
        van_der_corput, numbers_gpu % (a+1)**(m))] + (numbers_gpu // (a+1)**(m))*(a+1)**(m)
    # permutation_gpu[cp.searchsorted(
    # ALL_THIRTY_TWO_DIGIT_NUMBERS_GPU[:FIBONACCI[m+3]], numbers_gpu % 2**(m+1))] + (numbers_gpu // 2**(m+1))*2**(m+1)
    numbers = cp.asnumpy(numbers_gpu)
    return numbers


def scramble_last_digit_general(numbers, m, a, b):

    numbers_gpu = cp.asarray(numbers)

    length_1 = van_der_corput_ending_less_than_a(m-1, a, b)
    length_2 = num_whole_numbers(m-1, a, b) - \
        van_der_corput_ending_less_than_a(m-1, a, b)
    # print(f'length_1={length_1} : length_2={length_2}')

    random_0_to_a = cp.array([cp.random.permutation(a+1)
                             for _ in range(length_1)], dtype='uint32')
    # array_a = cp.full((length_1, 1), a,  dtype='uint32')
    # random_0_to_a = cp.append(random_0_to_a, array_a, axis=1)
    # print('random_0_to_a')
    # print(random_0_to_a)
    random_0_to_a = cp.transpose(random_0_to_a).reshape(
        np.product(random_0_to_a.shape))
    # print(random_0_to_a)

    random_0_to_b = cp.array([cp.random.permutation(b)
                             for _ in range(length_2)], dtype='uint32')
    # print('random_0_b')
    # print(random_0_to_b)
    random_0_to_b = cp.transpose(random_0_to_b).reshape(
        np.product(random_0_to_b.shape))
    # print(random_0_to_b)

    permutation_gpu = cp.zeros(num_whole_numbers(m, a, b), dtype='int32')
    van_der_corput = cp.asarray(van_der_corput_numbers(m, a, b))
    multiplier = (a+1)**(m-1)
    length_1 = num_whole_numbers(m-1, a, b)
    length_2 = van_der_corput_ending_less_than_a(m-1, a, b)
    length_3 = length_1-length_2
    # print(f'length_1={length_1} : length_2={length_2} : length_3={length_3}')

    # print(van_der_corput_one_less)
    # previous digit is not a
    for j in range(b):
        # print(random_0_to_a[j*length_2:(j+1)*length_2])
        permutation_gpu[j*length_1: j*length_1 +
                        length_2] += random_0_to_a[j*length_2:(j+1)*length_2]*multiplier + van_der_corput[:length_2]
    for j in range(a-b+1):
        permutation_gpu[b*length_1+j*length_2:b*length_1 +
                        (j+1)*length_2] += random_0_to_a[b*length_2 + j*length_2: b*length_2 + (j+1)*length_2]*multiplier + van_der_corput[:length_2]
    # previous digit is a
    for j in range(b):
        permutation_gpu[j*length_1 + length_2:(
            j+1)*length_1] += van_der_corput[length_2:length_1] + random_0_to_b[j*length_3:(j+1)*length_3]*multiplier

    # for j in range(b):
    #     # print(random_0_to_a[j*length_2:(j+1)*length_2])
    #     permutation_gpu[j*length_1: j*length_1 +
    #                     length_2] += random_0_to_a[j*length_2:(j+1)*length_2]*multiplier + ALL_VAN_DER_CORPUT_GPU[:length_2]
    # for j in range(a-b+1):
    #     permutation_gpu[b*length_1+j*length_2:b*length_1 +
    #                     (j+1)*length_2] += random_0_to_a[b*length_2 + j*length_2: b*length_2 + (j+1)*length_2]*multiplier + ALL_VAN_DER_CORPUT_GPU[:length_2]
    # # previous digit is a
    # for j in range(b):
    #     permutation_gpu[j*length_1 + length_2:(
    #         j+1)*length_1] += ALL_VAN_DER_CORPUT_GPU[length_2:length_1] + random_0_to_b[j*length_3:(j+1)*length_3]*multiplier

    # print(permutation_gpu)
    # print(van_der_corput_numbers(m, a, b))
    # print(numbers_to_digits(cp.asnumpy(permutation_gpu), m, a))
    # van_der_corput_all = cp.asarray(van_der_corput_numbers(m, a, b))

    # print(f'{permutation_gpu % (a+1)**(m)}')
    # print(van_der_corput_all)
    numbers_gpu = permutation_gpu[cp.searchsorted(
        van_der_corput, numbers_gpu % (a+1)**(m))] + (numbers_gpu // (a+1)**(m))*(a+1)**(m)
    # permutation_gpu[cp.searchsorted(
    # ALL_THIRTY_TWO_DIGIT_NUMBERS_GPU[:FIBONACCI[m+3]], numbers_gpu % 2**(m+1))] + (numbers_gpu // 2**(m+1))*2**(m+1)
    numbers = cp.asnumpy(numbers_gpu)
    return numbers


# @njit()
def scramble_non_zero_point_digits(point_numbers, num_digits, a, b):
    # n = num_digits
    for m in range(1, num_digits+1):
        point_numbers[:, 0] = scramble_mth_digit_general(
            point_numbers[:, 0], m, a, b)
    for m in range(1, num_digits+1):
        point_numbers[:, 1] = scramble_mth_digit_general(
            point_numbers[:, 1], m, a, b)
    return point_numbers


# @njit()
def scramble_non_zero_point_digits_extra_end(point_numbers, num_digits, a, b):
    # n = num_digits
    for m in range(1, num_digits):
        point_numbers[:, 0] = scramble_mth_digit_general(
            point_numbers[:, 0], m, a, b)
    point_numbers[:, 0] = scramble_last_digit_general(
        point_numbers[:, 0], num_digits, a, b)

    for m in range(1, num_digits):
        point_numbers[:, 1] = scramble_mth_digit_general(
            point_numbers[:, 1], m, a, b)
    point_numbers[:, 1] = scramble_last_digit_general(
        point_numbers[:, 1], num_digits, a, b)
    return point_numbers

# @njit()


def scrambled_hammersley_set(num_digits, a, b):
    point_numbers = hammersley_numbers(num_digits, a, b)
    scramble_non_zero_point_digits(point_numbers, num_digits, a, b)

    point_values = points_to_values(point_numbers, num_digits, a, b)
    base = ((math.sqrt(a ** 2+4*b)+a) / 2).real
    point_numbers //= a*(a+1)**(num_digits-1)
    # print(point_numbers)
    rand = np.random.uniform(0, 1, point_values.shape)
    point_values += rand*point_numbers * b*base**(-num_digits-1)
    # print(1-point_numbers)
    # print(rand*(1-point_numbers) * (a) * base**(-num_digits-1))
    point_values += rand*(1-point_numbers) * base**(-num_digits)
    return point_values


def scrambled_hammersley_set_extra_end(num_digits, a, b):
    point_numbers = hammersley_numbers(num_digits, a, b)
    scramble_non_zero_point_digits_extra_end(point_numbers, num_digits, a, b)

    point_values = points_to_values(point_numbers, num_digits, a, b)
    base = ((math.sqrt(a ** 2+4*b)+a) / 2).real
    point_numbers //= a*(a+1)**(num_digits-1)
    # print(point_numbers)
    rand = np.random.uniform(0, 1, point_values.shape)
    point_values += rand*point_numbers * b*base**(-num_digits-1)
    # print(1-point_numbers)
    # print(rand*(1-point_numbers) * (a) * base**(-num_digits-1))
    point_values += rand*(1-point_numbers) * base**(-num_digits)
    return point_values


# # @njit()
# def star_discrepancy_1_d(arr):
#     points_to_test = np.sort(np.copy(arr))
#     n = points_to_test.shape[0]
#     to_take_max = np.array([points_to_test[k]-(2*k-1)/(2*n) for k in range(n)])
#     return 1/(2*n) + np.max(to_take_max)


# def single_discrepancy_one_dimension(values, n_index, lambda_index):
#     bool = points <= points[point_index]
#     num_points_inside = np.sum(bool)
#     return num_points_inside / points.shape[0] - points[point_index]


# @njit()
# def single_discrepancy_two_dimensions(values, points, n_index, lambda_index):
#     for point in points:
#         if point[0] < point
#     num_points_inside = np.sum(bool)
#     return num_points_inside / points.shape[0] - points[point_index]


# @njit()
def test_transfer_discrepancy(num_digits, a, b, lambda_index, n_index):
    # first coord is vdc second coord is nth whole number / gamma^m
    points = np.flip(hammersley_values_2(num_digits, a, b), axis=1)

    # print(points[:10])

    vdc = points[:n_index, 0]
    bool = vdc < points[lambda_index, 1]
    num_points_inside = np.sum(bool)
    rhs = num_points_inside / vdc.shape[0] - points[lambda_index, 1]

    point_count = 0
    for point in points:
        if point[0] < points[lambda_index, 1] and point[1] < points[n_index, 1]:
            point_count += 1
    lhs = point_count / points.shape[0] - \
        points[lambda_index, 1] * points[n_index, 1]

    print(f'{lhs == rhs} : {lhs} : {rhs}')


# num_digits = 10
# a = 2
# b = 1
# values = van_der_corput_values(num_digits, a, b)

# # n_index = 150
# # lambda_index = 300
# # print(single_discrepancy_one_dimension(values[:n_index], point_index))

# # points = np.flip(hammersley_values_2(num_digits, a, b), axis=1)
# # print(single_discrepancy_two_dimensions(values,points,n_index,lambda_index))

# test_transfer_discrepancy(num_digits, a, b, 100, 200)

a = 2
b = 1
for n in range(0, 10):
    # print(f'{((1+math.sqrt(2).real)/2)**n}  : {num_whole_numbers_2(n, a, b) / (1+math.sqrt(2).real)**n}')
    print(f' {((1+math.sqrt(2).real)**(n+1) + (1-math.sqrt(2).real)**(n+1))/2 }  : {num_whole_numbers_2(n, a, b)}')
