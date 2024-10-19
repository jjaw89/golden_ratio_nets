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
    p_minus_2 = 0
    p_minus_1 = 1
    for _ in range(n+1):
        p = 2*p_minus_1 + p_minus_2
        p_minus_2 = p_minus_1
        p_minus_1 = p

    return p


# for n in range(3):
#     print(f'{partial_quotients(n)} : {base_3_numbers(n).shape[0]}')

# for n in range(-3, 3):
#     print(f'{n} : {SQRT_2_UNIT**n}')


# print()
# # print()
# num_digits = 3
# a = digit_expansion(num_digits)
# a[np.lexsort(np.transpose(a)[::-1])]
# print(a[np.lexsort(np.transpose(a)[::-1])])
# print(partition(3))
# print(van_der_corput_2(3))
# print(sqrt_2_unit_hammersley(num_digits))

# for num_digits in range(12, 14):
#     plot_points(sqrt_2_unit_hammersley(num_digits), 0, 0, 1)


# for num_digits in range(1, 12):
#     print(
#         f'{num_digits} digits : {base_3_numbers(num_digits).shape[0]} points')

# num_digits = 3
# print(digit_expansion(num_digits))
# print()
# a = digit_expansion(num_digits)
# b = base_3_numbers(num_digits)
# numbers = a[np.lexsort(np.transpose(a)[::-1])]
# print(numbers)
# print(np.flip(numbers, axis=1))

# print(van_der_corput_2(num_digits))
# # print()
# # print(increasing_digits(num_digits))
# # # print(sqrt_2_unit_hammersley(3))
# plot_points(sqrt_2_unit_hammersley(0), 1, 0, 10)
# plot_points(sqrt_2_unit_hammersley(0), 2, 0, 10)
# plot_points(sqrt_2_unit_hammersley(0), 3, 0, 10)

print(digit_expansion(3))

a = digit_expansion(3)
numbers = a[np.lexsort(np.transpose(a)[::-1])]
print(np.flip(numbers, axis=1))
