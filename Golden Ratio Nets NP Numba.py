
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


import matplotlib.pyplot as plt
from sympy import GoldenRatio, plot
plt.style.use('seaborn-whitegrid')

GOLDEN_RATIO = ((1+sqrt(5))/2).real


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


# start_time = time.perf_counter()
# vectors = no_consecutive_ones(num_digits)
# print(vectors)
# print(vectors.shape)
# end_time = time.perf_counter()
# print(sys.getsizeof(vectors))
# print(f"Execution Time : {end_time-start_time:0.6f}")

# start_time = time.perf_counter()
# vectors = no_consecutive_ones_b(num_digits)
# print(vectors)
# print(vectors.shape)
# print(sys.getsizeof(vectors))
# end_time = time.perf_counter()

# num_digits = 32

# start_time = time.perf_counter()
# print(points(num_digits))
# end_time = time.perf_counter()
# print(f"Execution Time : {end_time-start_time:0.6f}")

# print(f"Number of Digits : {num_digits}")

# start_time = time.perf_counter()
# zeros_and_ones = no_consecutive_ones_b(
#     num_digits)
# end_time = time.perf_counter()
# print(f"Zeros and Ones Time : {end_time-start_time:0.6f}")

# start_time = time.perf_counter()
# points = np.matmul(zeros_and_ones, golden_multiplied_2d(num_digits))
# end_time = time.perf_counter()
# print(f"Matrix Multiplication Time : {end_time-start_time:0.6f}")


# start_time = time.perf_counter()
# points = points[points[:, 0].argsort()]
# end_time = time.perf_counter()

# print(points.shape)

# print(f"Sort Time : {end_time-start_time:0.6f}")


# print(num_digits)
# start_time = time.perf_counter()
# partition(num_digits)
# end_time = time.perf_counter()

# print(f"Total Time : {end_time-start_time:0.6f}")


# x = np.array([1, 2, 3])
# y = np.array([1, 2, 3])

# plt.plot(x, y, 'o', color='black')

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


# num_digits = 4
# x_partition_num_digits = 0
# y_partition_num_digits = 1


# hammersley_points = points(num_digits)
# print(hammersley_points)

# x_partition = partition(x_partition_num_digits)
# print(x_partition)
# x_axis_bins = [[] for i in range(x_partition.size-1)]
# # print(x_axis_bins)

# bin_number = 0
# for point in list(hammersley_points):
#     if point[0] > x_partition[bin_number + 1]:
#         bin_number = bin_number + 1
#     else:
#         x_axis_bins[bin_number].append(list(point))

# for bin in x_axis_bins:
#     bin.sort(key=lambda x: x[1])


# y_partition = partition(y_partition_num_digits)
# y_axis_bins = [[[] for j in range(y_partition.size-1)]
#                for i in range(x_partition.size-1)]
# print(y_partition)
# # print(y_axis_bins)

# for x_bin_number in range(len(x_axis_bins)):
#     y_bin_number = 0
#     for point in x_axis_bins[x_bin_number]:
#         if point[1] > y_partition[y_bin_number + 1]:
#             y_bin_number = y_bin_number + 1
#         else:
#             y_axis_bins[x_bin_number][y_bin_number].append(list(point))

# print(hammersley_points.shape)
# print(GOLDEN_RATIO)
# print(hammersley_points)
# print()
# print(x_axis_bins)
# print()
# print(y_axis_bins)

# num_digits = 3
# vectors = no_consecutive_ones_b(num_digits)
# vectors_1 = vectors[np.lexsort(np.rot90(vectors))]
# print(vectors_1)
# vectors = no_consecutive_ones_b(num_digits+3)
# vectors_2 = vectors[np.lexsort(np.rot90(vectors))]
# print(vectors_2)
# # print(np.prod(vectors[0] <= vectors[2]))

# num_digits = 6
# plot_hammersley(num_digits, 5, 0)
# num_digits = 5
# print(np.array([[scipy.special.binom(i, j)
#                  for i in range(0, num_digits)] for j in range(0, num_digits)]))

# print()
# matrix = np.array([[scipy.special.binom(i, j) % 2
#                     for i in range(0, num_digits)] for j in range(0, num_digits)])

# matrix = np.array([
#     [1., 0., 1., 0., 1.],
#     [0., 1., 0., 1., 0.],
#     [0., 0., 1., 0., 0.],
#     [0., 0., 0., 1., 0.],
#     [0., 0., 0., 0., 1.]])


# print(matrix)
# print()
# print(vectors)
# print()
# multiplied = np.matmul(matrix, vectors)

# print(multiplied % 2)


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


def van_der_corput(num_digits):
    vectors = no_consecutive_ones_b(num_digits)
    print(vectors)
    vectors = vectors[np.lexsort(np.rot90(vectors))]
    print(vectors)
    print()
    x_coords = np.matmul(vectors, np.flip(
        golden_multiplied_1d(num_digits))).real
    print(x_coords)
    return


# vectors = no_consecutive_ones_b(5)
# vectors = vectors[np.lexsort(np.rot90(vectors))]
# print(np.fliplr(vectors))
# print(vectors)

# plot_hammersley(6, 4, 3)
# # van_der_corput(6)


def new_recursive_d(a, b, d):
    n = a.shape[0]
    m = b.shape[0]

    if m == 0:
        return 0
    elif d == 0:

        # print()
        # print(f"a : {a}")
        # print(f"b : {b}")
        # print(np.sum(a) * np.sum(b))

        return np.sum(a) * np.sum(b)
    elif n == 1:
        # This might not be the fastest way
        if d == 1:
            # min_array = np.array(
            #     [b[index_j, 1] * min(a[0, 0], b[index_j, 0]) for index_j in range(m)])

            # print()
            # print(f"a : {a}")
            # print(f"b : {b}")
            # print(min_array)
            # print(a[0, -1] * np.sum(min_array))

            # return a[0, -1] * np.sum(min_array)

            partial_sum = 0
            for index_j in range(m):
                partial_sum = partial_sum + \
                    b[index_j, 1] * min(a[0, 0], b[index_j, 0])
            return a[0, -1] * partial_sum

        else:
            # min_array = np.array([b[index_j, 2] * min(a[0, 0], b[index_j, 0])
            #                      * min(a[0, 1], b[index_j, 1]) for index_j in range(m)])

            # # print()
            # # print(f"a : {a}")
            # # print(f"b : {b}")
            # # print(min_array)
            # # print(a[0, -1] * np.sum(min_array))

            # return a[0, -1] * np.sum(min_array)

            partial_sum = 0
            for index_j in range(m):
                partial_sum = partial_sum + \
                    b[index_j, 2] * min(a[0, 0], b[index_j, 0]) * \
                    min(a[0, 1], b[index_j, 1])
            return a[0, -1] * partial_sum
    else:
        mu = np.median(a[:, d-1])

        bool_arr = a[:, d-1] <= mu

        a_l = a[bool_arr]
        a_r = a[np.logical_not(bool_arr)]

        bool_arr = b[:, d-1] <= mu

        b_l = b[bool_arr]
        b_r = b[np.logical_not(bool_arr)]

        a_l_tilde = a_l
        a_r_bar = a_r

        b_l_tilde = b_l
        b_r_bar = b_r

        a_l_tilde = np.delete(a_l_tilde, d-1, 1)
        a_r_bar = np.delete(a_r_bar, d-1, 1)
        b_l_tilde = np.delete(b_l_tilde, d-1, 1)
        b_r_bar = np.delete(b_r_bar, d-1, 1)

        a_l_tilde_last_column = a_l[:, -1]*a_l[:, -2]
        b_l_tilde_last_column = b_l[:, -1]*b_l[:, -2]

        a_l_tilde[:, -1] = a_l_tilde_last_column
        b_l_tilde[:, -1] = b_l_tilde_last_column

        # print()
        # print()

        # print(f"a : {a}")
        # print(f"b : {b}")

        # print(f"mu : {mu}")

        # print(f"a_l : {a_l}")
        # print(f"b_l : {b_l}")

        # print(f"a_r : {a_r}")
        # print(f"b_r : {b_r}")

        # print(f"a_l_tilde : {a_l_tilde}")
        # print(f"b_l_tilde : {b_l_tilde}")

        # print(f"a_r_bar : {a_r_bar}")
        # print(f"b_r_bar : {b_r_bar}")

        # print()
        # print()

        case1 = new_recursive_d(a_l, b_l, d)
        case2 = new_recursive_d(a_r, b_r, d)
        case3 = new_recursive_d(a_l_tilde, b_r_bar, d-1)
        case4 = new_recursive_d(a_r_bar, b_l_tilde, d-1)

        return case1 + case2 + case3 + case4


def recursive_d(a, b, d):
    n = a.shape[0]
    m = b.shape[0]

    if m == 0:
        return 0
    elif d == 0:

        # print()
        # print(f"a : {a}")
        # print(f"b : {b}")
        # print(np.sum(a) * np.sum(b))

        return d_is_0(a, b)
    elif n == 1:
        # This might not be the fastest way
        if d == 1:
            # min_array = np.array(
            #     [b[index_j, 1] * min(a[0, 0], b[index_j, 0]) for index_j in range(m)])

            # print()
            # print(f"a : {a}")
            # print(f"b : {b}")
            # print(min_array)
            # print(a[0, -1] * np.sum(min_array))

            # return a[0, -1] * np.sum(min_array)

            return d_is_1(a, b, m)

        else:
            # min_array = np.array([b[index_j, 2] * min(a[0, 0], b[index_j, 0])
            #                      * min(a[0, 1], b[index_j, 1]) for index_j in range(m)])

            # # print()
            # # print(f"a : {a}")
            # # print(f"b : {b}")
            # # print(min_array)
            # # print(a[0, -1] * np.sum(min_array))

            # return a[0, -1] * np.sum(min_array)

            return d_is_2(a, b, m)
    else:
        mu = np.median(a[:, d-1])

        bool_arr = a[:, d-1] <= mu

        a_l = a[bool_arr]
        a_r = a[np.logical_not(bool_arr)]

        bool_arr = b[:, d-1] <= mu

        b_l = b[bool_arr]
        b_r = b[np.logical_not(bool_arr)]

        a_l_tilde = a_l
        a_r_bar = a_r

        b_l_tilde = b_l
        b_r_bar = b_r

        a_l_tilde = np.delete(a_l_tilde, d-1, 1)
        a_r_bar = np.delete(a_r_bar, d-1, 1)
        b_l_tilde = np.delete(b_l_tilde, d-1, 1)
        b_r_bar = np.delete(b_r_bar, d-1, 1)

        a_l_tilde_last_column = a_l[:, -1]*a_l[:, -2]
        b_l_tilde_last_column = b_l[:, -1]*b_l[:, -2]

        a_l_tilde[:, -1] = a_l_tilde_last_column
        b_l_tilde[:, -1] = b_l_tilde_last_column

        # print()
        # print()

        # print(f"a : {a}")
        # print(f"b : {b}")

        # print(f"mu : {mu}")

        # print(f"a_l : {a_l}")
        # print(f"b_l : {b_l}")

        # print(f"a_r : {a_r}")
        # print(f"b_r : {b_r}")

        # print(f"a_l_tilde : {a_l_tilde}")
        # print(f"b_l_tilde : {b_l_tilde}")

        # print(f"a_r_bar : {a_r_bar}")
        # print(f"b_r_bar : {b_r_bar}")

        # print()
        # print()
        return recursive_d(a_l, b_l, d)+recursive_d(a_r, b_r, d)+recursive_d(a_l_tilde, b_r_bar, d-1)+recursive_d(a_r_bar, b_l_tilde, d-1)


@njit()
def d_is_0(a, b):
    return np.sum(a)*np.sum(b)


@njit()
def d_is_1(a, b, m):
    partial_sum = 0
    for index_j in prange(m):
        partial_sum = partial_sum + \
            b[index_j, 1] * min(a[0, 0], b[index_j, 0])
    return a[0, -1] * partial_sum


@njit()
def d_is_2(a, b, m):
    partial_sum = 0
    for index_j in prange(m):
        partial_sum = partial_sum + \
            b[index_j, 2] * min(a[0, 0], b[index_j, 0]) * \
            min(a[0, 1], b[index_j, 1])
    return a[0, -1] * partial_sum


def function_d(arr):
    var_sum = 0
    # print()
    for x in arr:
        for y in arr:
            prod_of_min = min(x[0], y[0])*min(x[1], y[1])

            # print()
            # print(f"x : {x}")
            # print(f"y : {y}")
            # print(
            #     f"min(x[0], y[0]) : {min(x[0], y[0])}, min(x[1], y[1]) : {min(x[1], y[1])}")
            # print(f"prod_of_min : {prod_of_min}")

            var_sum = var_sum + prod_of_min
    return var_sum


@njit(parallel=True)
def parallel_d(arr):
    var_sum = 0
    # print()
    for x in prange(arr.shape[0]):
        for y in prange(arr.shape[0]):

            # print()
            # print(f"x : {x}")
            # print(f"y : {y}")
            # print(
            #     f"min(x[0], y[0]) : {min(x[0], y[0])}, min(x[1], y[1]) : {min(x[1], y[1])}")
            # print(f"prod_of_min : {prod_of_min}")

            var_sum += min(arr[x, 0], arr[y, 0])*min(arr[x, 1], arr[y, 1])
    return var_sum


function_d_njit = njit()(function_d)
# function_d_parallel_njit = njit(parallel=True)(function_d_parallel)


# print(point_set)


# start_time = time.perf_counter()
# print(f"Result function_d_jit : {function_d_njit(point_set)}")
# end_time = time.perf_counter()
# time_1_njit = end_time-start_time
# print(f"Standard Time njit : {time_1_njit:0.6f}")
# print()

# start_time = time.perf_counter()
# print(f"Result function_d_jit : {new_function_d_njit(point_set)}")
# end_time = time.perf_counter()
# new_time_1_njit = end_time-start_time
# print(f"new Standard Time njit : {new_time_1_njit:0.6f}")
# print()

function_d_njit(points(1))
parallel_d(points(1))

num_digits = 5
point_set = points(num_digits)
print(point_set.shape)
point_set_ones = np.append(point_set, np.ones((point_set.shape[0], 1)), axis=1)
recursive_d(point_set_ones, point_set_ones, 2)

num_digits = 30
point_set = points(num_digits)
print(point_set.shape)
point_set_ones = np.append(point_set, np.ones((point_set.shape[0], 1)), axis=1)

start_time = time.perf_counter()
print(f"Result parallel_d : {parallel_d(point_set)}")
end_time = time.perf_counter()
time_2 = end_time-start_time
print(f"Recursion Time : {time_2:0.6f}")


# start_time = time.perf_counter()
# print(f"function_d_jit : {function_d_njit(point_set)}")
# end_time = time.perf_counter()
# time_1 = end_time-start_time
# print(f"Recursion Time : {time_1:0.6f}")

# print()
# print(f"Normal/Parallel : {time_1/time_2}")

# print(point_set)
# print(np.ones((point_set.shape[0], 1)))

# print(point_set_ones)

# num_digits = 10
# point_set = points(num_digits)

# start_time = time.perf_counter()
# print(function_d(point_set))
# end_time = time.perf_counter()
# time_2 = end_time-start_time
# print(num_digits)
# print(f"Normal Time : {time_2:0.6f}")


# start_time = time.perf_counter()
# print(new_recursive_d(point_set_ones, point_set_ones, 2))
# end_time = time.perf_counter()
# time_1 = end_time-start_time
# print(f"Recursion Time : {time_1:0.6f}")


start_time = time.perf_counter()
print(recursive_d(point_set_ones, point_set_ones, 2))
end_time = time.perf_counter()
time_1 = end_time-start_time
print(f"Recursion Time jit : {time_1:0.6f}")


print(f"Recursion/Normal: {time_1/time_2}")
