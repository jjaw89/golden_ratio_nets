
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

REDUCTION_VECTOR = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
                            0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype='int8')


def no_consecutive_ones(length):
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
    vectors = np.append(end_in_one, end_in_zero, axis=0)
    return vectors[np.lexsort(np.rot90(vectors))]


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
    # print("binary vectors")
    # print(vectors)
    return vectors[np.lexsort(np.rot90(vectors))]


def binary_vectors_ending_in_one(length):
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

    return vectors_one


def extend_matrix_vector(starting_matricies):
    zeros = np.zeros(
        (starting_matricies.shape[0], 1, starting_matricies.shape[2]), dtype='b')
    print(zeros)
    matricies = np.append(starting_matricies, zeros, axis=1)
    # print(matricies.shape)
    binary = binary_vectors_ending_in_one(matricies.shape[1])
    binary = np.reshape(binary, (binary.shape[0], binary.shape[1], 1))
    # print(binary.shape)
    repeated_binary = binary.copy()
    # print(repeated_binary.shape)
    for _ in range(matricies.shape[0]-1):
        repeated_binary = np.append(repeated_binary, binary, axis=0)
    # print(repeated_binary.shape)
    matricies = np.repeat(
        matricies, 2 ** (matricies.shape[1]-1), axis=0)
    # print(matricies.shape)
    return np.append(matricies, repeated_binary, axis=2)


def extend_matrix_vector_all(starting_matricies):
    binary = binary_vectors(starting_matricies.shape[1])
    binary = np.reshape(binary, (binary.shape[0], 1, binary.shape[1]))
    print(binary)
    # print(zeros)
    # print(matricies.shape)
    repeated_binary = binary.copy()
    # print(repeated_binary.shape)
    for _ in range(starting_matricies.shape[0]-1):
        repeated_binary = np.append(repeated_binary, binary, axis=0)
    print(repeated_binary)
    print(repeated_binary.shape)
    matricies = np.repeat(starting_matricies, 2 **
                          (starting_matricies.shape[1]), axis=0)
    print(matricies)

    matricies = np.append(matricies, repeated_binary, axis=1)

    print(matricies)
    print(matricies.shape[1])
    binary = binary_vectors(matricies.shape[1])
    binary = np.reshape(binary, (binary.shape[0], binary.shape[1], 1))
    print(matricies.shape)
    print(binary.shape)
    repeated_binary = binary.copy()
    # print(repeated_binary.shape)
    for _ in range(matricies.shape[0]-1):
        repeated_binary = np.append(repeated_binary, binary, axis=0)
    # print(repeated_binary.shape)
    matricies = np.repeat(
        matricies, 2 ** (matricies.shape[1]), axis=0)
    # print(matricies.shape)
    return np.append(matricies, repeated_binary, axis=2)


def reduction(x):
    return REDUCTION_VECTOR[x]


def reduction_2(x):
    return x % 2
# start_time = time.perf_counter()

# matrix_vector = np.array(
#     [[[1, 0], [0, 1]], [[1, 1], [0, 1]]], dtype='b')
# new_matricies = extend_matrix_vector(matrix_vector)

# end_time = time.perf_counter()
# time_difference = end_time-start_time
# print(f"Time : {time_difference :0.6f}")


# def test_extended_matricies(matrix_vector):
#     new_matricies = extend_matrix_vector(matrix_vector)
#     vectors = no_consecutive_ones(new_matricies.shape[1])
#     # print(vectors)
#     valid_matricies = []
#     for matrix in new_matricies:
#         # print(matrix)
#         result = reduction(
#             np.matmul(matrix, np.transpose(vectors)))
#         result = np.transpose(result)
#         # print(result)
#         # print(result[np.lexsort(np.rot90(result))])
#         # print(np.array_equal(vectors, result[np.lexsort(np.rot90(result))]))
#         if np.array_equal(vectors, result[np.lexsort(np.rot90(result))]):
#             valid_matricies += [matrix.tolist()]
#         # print(valid_matricies)
#     return np.array(valid_matricies)


def test_extended_matricies(matrix_vector):
    new_matricies = extend_matrix_vector(matrix_vector)
    vectors = no_consecutive_ones(new_matricies.shape[1])

    print(vectors)
    valid_matricies = []
    for matrix in new_matricies:
        print("---------------------------")
        print(matrix)
        result = np.matmul(matrix, np.transpose(vectors))
        print()
        print(result)
        result = reduction(result)
        result = np.transpose(result)
        # print(result)
        # print(result[np.lexsort(np.rot90(result))])
        print(np.array_equal(vectors, result[np.lexsort(np.rot90(result))]))
        if np.array_equal(vectors, result[np.lexsort(np.rot90(result))]):
            valid_matricies += [matrix.tolist()]
        # print(valid_matricies)
    return np.array(valid_matricies)


matrix_vector = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 1]]], dtype='b')

# extend_matrix_vector(matrix_vector)
print(extend_matrix_vector_all(matrix_vector))
# matrix_vector = test_extended_matricies(matrix_vector)
# print("+++++++++++++++++++++++++++++++")
# matrix_vector = test_extended_matricies(matrix_vector)


def all_possible_matricies():
    rows = binary_vectors(3)
    matricies = np.array([[rows[x], rows[y], rows[z]] for x in range(1,
                                                                     2 ** 3) for y in range(1, 2 ** 3) for z in range(1, 2 ** 3)])
    # for x in range(2^dim):
    #     for y in range(2^dim):
    #         for z in range(2^dim)
    return matricies


def test_all_3x3_matricies():
    new_matricies = all_possible_matricies()
    vectors = no_consecutive_ones(new_matricies.shape[1])
    # print(vectors)
    valid_matricies = []
    for matrix in new_matricies:
        # print(matrix)
        result = np.matmul(matrix, np.transpose(vectors))
        # print(result)
        result = reduction(result)
        result = np.transpose(result)
        # print(result)
        # print(result[np.lexsort(np.rot90(result))])
        # print(np.array_equal(vectors, result[np.lexsort(np.rot90(result))]))
        if np.array_equal(vectors, result[np.lexsort(np.rot90(result))]):
            valid_matricies += [matrix.tolist()]
        # print(valid_matricies)
    return np.array(valid_matricies)


# print(test_all_3x3_matricies())


def all_possible_4x4_matricies():
    rows = binary_vectors(4)
    matricies = np.array([[rows[x], rows[y], rows[z], rows[w]] for w in range(1, 2**4) for x in range(1,
                                                                                                      2 ** 4) for y in range(1, 2 ** 4) for z in range(1, 2 ** 4)])
    # for x in range(2^dim):
    #     for y in range(2^dim):
    #         for z in range(2^dim)
    return matricies


def test_all_4x4_matricies():
    new_matricies = all_possible_4x4_matricies()
    vectors = no_consecutive_ones(new_matricies.shape[1])
    # print(vectors)
    valid_matricies = []
    for matrix in new_matricies:
        # print(matrix)
        result = np.matmul(matrix, np.transpose(vectors))
        # print(result)
        result = reduction(result)
        result = np.transpose(result)
        # print(result)
        # print(result[np.lexsort(np.rot90(result))])
        # print(np.array_equal(vectors, result[np.lexsort(np.rot90(result))]))
        if np.array_equal(vectors, result[np.lexsort(np.rot90(result))]):
            valid_matricies += [matrix.tolist()]
        # print(valid_matricies)
    return np.array(valid_matricies)


# print(test_all_4x4_matricies())


def all_possible_5x5_matricies():
    rows = binary_vectors(5)
    matricies = np.array([[rows[x], rows[y], rows[z], rows[w], rows[k]] for k in range(1, 2**5) for w in range(1, 2**5) for x in range(1,
                                                                                                                                       2 ** 5) for y in range(1, 2 ** 5) for z in range(1, 2 ** 5)])
    # for x in range(2^dim):
    #     for y in range(2^dim):
    #         for z in range(2^dim)
    return matricies


def test_all_5x5_matricies():
    new_matricies = all_possible_5x5_matricies()
    vectors = no_consecutive_ones(new_matricies.shape[1])
    # print(vectors)
    valid_matricies = []
    for matrix in new_matricies:
        # print(matrix)
        result = np.matmul(matrix, np.transpose(vectors))
        # print(result)
        result = reduction_2(result)
        result = np.transpose(result)
        # print(result)
        # print(result[np.lexsort(np.rot90(result))])
        # print(np.array_equal(vectors, result[np.lexsort(np.rot90(result))]))
        if np.array_equal(vectors, result[np.lexsort(np.rot90(result))]):
            valid_matricies += [matrix.tolist()]
        # print(valid_matricies)
    return np.array(valid_matricies)


# print(test_all_5x5_matricies())
