from cmath import sqrt
from dataclasses import replace
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
import random
from sympy import GoldenRatio, plot
import timeit
import cupy as cp
plt.style.use('seaborn-whitegrid')

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
SQRT_2_UNIT = (1+sqrt(2)).real


@njit()
def base_3_numbers(num_digits):
    if num_digits == 0:
        return np.array([0], dtype='int64')

    end_in_0 = np.array([0], dtype='int64')
    end_in_1 = np.array([1], dtype='int64')
    end_in_2 = np.array([2], dtype='int64')

    for m in range(1, num_digits):
        next_end_in_1 = np.append(end_in_0, end_in_1, axis=0)
        next_end_in_0 = np.append(next_end_in_1, end_in_2, axis=0)

        end_in_2 = np.copy(end_in_0) + 2*3**m
        end_in_0 = np.copy(next_end_in_0)
        end_in_1 = np.copy(next_end_in_0) + 3**m

    final_numbers = np.append(end_in_0, end_in_1, axis=0)
    final_numbers = np.append(final_numbers, end_in_2, axis=0)
    return final_numbers


def van_der_corput(num_digits):
    # digits = np.lexsort(digit_expansion(num_digits), axis=0)
    digits = base_3_numbers(num_digits)
    values = np.zeros(digits.shape[0], dtype='float')
    for n in range(digits.shape[0]):
        for m in range(num_digits):
            values[n] += digits[n][m]*SQRT_2_UNIT ** (-num_digits+m)
    return values


def partition(num_digits):
    numbers = base_3_numbers(num_digits)
    values = np.zeros_like(numbers, dtype='float')
    for n in range(numbers.shape[0]):
        k = numbers[n]
        for m in range(1, num_digits+1):
            digit = k % 3
            values[n] += digit * SQRT_2_UNIT ** (-num_digits-1+m)
            k //= 3
    return values


def van_der_corput_2(num_digits):
    a = digit_expansion(num_digits)
    b = base_3_numbers(num_digits)
    numbers = b[np.lexsort(np.transpose(a)[::-1])]
    values = np.zeros_like(numbers, dtype='float')
    for n in range(numbers.shape[0]):
        k = numbers[n]
        for m in range(1, num_digits+1):
            digit = k % 3
            values[n] += digit * SQRT_2_UNIT ** (-num_digits-1+m)
            k //= 3
    return values


def sqrt_2_unit_hammersley(num_digits):
    # x = partition(num_digits)
    x = partition(num_digits)
    y = van_der_corput_2(num_digits)
    # print(x)
    return np.array([[x[n], y[n]] for n in range(x.shape[0])], dtype='float')


def digit_expansion(num_digits):
    numbers = base_3_numbers(num_digits)
    digits = np.zeros((numbers.shape[0], num_digits), dtype='int64')
    for n in range(numbers.shape[0]):
        k = numbers[n]
        for m in range(num_digits):
            digit = k % 3
            digits[n, m] = digit
            k //= 3
    return digits


def increasing_digits(num_digits):
    numbers = base_3_numbers(num_digits)
    values = np.zeros_like(numbers, dtype='float')
    for n in range(numbers.shape[0]):
        k = numbers[n]
        for m in range(num_digits):
            digit = k % 3
            values[n] += digit * SQRT_2_UNIT ** (m)
            k //= 3
    return values


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


def partial_quotients(n):
    p_minus_2 = 1
    p_minus_1 = 3
    if n == 1:
        return 3
    for _ in range(n-1):
        p = 2*p_minus_1 + p_minus_2
        p_minus_2 = p_minus_1
        p_minus_1 = p

    return p


def partial_quotients_2(n):
    p_minus_2 = 1
    p_minus_1 = 3
    if n == 1:
        return 3
    for _ in range(n-1):
        p = 2*p_minus_1 + 2*p_minus_2
        p_minus_2 = p_minus_1
        p_minus_1 = p

    return p

###########################################


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
def van_der_corput_golden_ratio(num_digits):
    numbers = no_consecutive_ones_binary_2(num_digits)
    decimals = convert_integers_to_decimals(numbers)
    return decimals

############################################


# print(partial_quotients(13))
# print(FIBONACCI[24+2])

# x = van_der_corput_golden_ratio(24)
# y = van_der_corput_2(13)

# halton_sequence = np.array([[x[n], y[n]]
#                            for n in range(partial_quotients(13))], dtype='float')


# for n in range(1, 14):

#     plot_points(halton_sequence[:partial_quotients(n)], 0, 0, 5)
#     plot_points(halton_sequence[:partial_quotients(n)], 0, 0, 1)

# for n in range(1, 26):

#     # plot_points(halton_sequence[:partial_quotients(n)], 0, 0, 5)
#     plot_points(halton_sequence[:FIBONACCI[n]], 0, 0, 1)

# plot_points(halton_sequence[:1000], 0, 0, 10)

for n in range(1, 20):
    print(f'{partial_quotients(n)}')

for n in range(1, 30):
    print(f'{FIBONACCI[n]}')
