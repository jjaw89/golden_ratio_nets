
from collections import Counter
from pathlib import Path
from ast import Break
from asyncio.windows_events import NULL
from cmath import sqrt
from copy import copy
import numpy as np
import time
import sys
import cupy as cp
import math
import scipy.special
from sqlalchemy import null
import os
import psutil
from numba import jit, njit, vectorize, prange, set_num_threads
from sys import getsizeof


n = 10
all_vectors = np.array([[x, y, z]
                        for x in range(0, n) for y in range(0, n) for z in range(0, n)], dtype='int16')
num_vectors = all_vectors.shape[0]
numbers = np.arange(1000, dtype='int16')


@njit()
def binary_vectors(length):
    end_in_zero = np.zeros((1, 1), dtype='bool')
    end_in_one = np.ones((1, 1), dtype='bool')

    vectors = np.append(end_in_zero, end_in_one, axis=0)

    for i in range(length-1):
        vectors_zero = np.append(vectors,
                                 np.zeros(
                                         (vectors.shape[0], 1), dtype='bool'),
                                 axis=1)
        vectors_one = np.append(vectors,
                                np.ones(
                                    (vectors.shape[0], 1), dtype='bool'),
                                axis=1)
        vectors = np.append(vectors_zero, vectors_one, axis=0)

    return vectors


@njit()
def compare(vector_1, vector_2):
    less_than = vector_1[0] <= vector_2[0]
    for i in range(1, 3):
        less_than = (vector_1[i] <= vector_2[i]) * less_than
    if less_than:
        return False
    greater_than = vector_1[0] >= vector_2[0]
    for i in range(1, 3):
        greater_than = (vector_1[i] >= vector_2[i]) * greater_than
    if greater_than:
        return False
    return True


# @njit()
def extend_by_one(array):
    if array.shape[0] == 1:
        to_be_added = compared[array[0]]
    else:
        to_be_added = compared[array[0]]
        for numb in array[1:]:
            to_be_added = np.multiply(to_be_added, compared[numb])

    # vect = numbers[to_be_added]
    # if vect.shape[0] == 0:
    #     return False
    # vect = np.vstack(vect, dtype='int16')

    # duplicated_row = np.array([array]*vect.shape[0])
    # # print(duplicated_row)
    # extended = np.append(duplicated_row, vect, axis=1)

    # to_be_added = np.prod(compared[array], axis=0,
    #                       dtype=np.bool_)  # check the axis
    numbers_to_add = numbers[to_be_added]
    if numbers_to_add.shape[0] == 0:
        return []
    else:
        starting_list = array.tolist()
        extended = [starting_list + [numbers_to_add[0]]]
        for numb in numbers_to_add[1:]:
            extended += [starting_list + [numb]]
    return extended


# print(extend_by_one(extended_antichains[1]))


def extend_array_1(starting_antichains):
    # print(starting_antichains.shape[0])
    for n in range(starting_antichains.shape[0]):
        if n % 1000 == 0:
            print(n)
        if len(extend_by_one(starting_antichains[n])) != 0:
            extend_antichains = extend_by_one(starting_antichains[n])
            break
    # print()
    # print(extend_antichains)
    # print(n)
    # print()
    for k in range(n+1, starting_antichains.shape[0]):
        extend_antichains += extend_by_one(starting_antichains[k])
        if k % 50000 == 0:
            print(k)

    return extend_antichains

# @njit()


def extend_array_2(starting_antichains):

    for n in range(starting_antichains.shape[0]):
        if extend_by_one(starting_antichains[n]).all():
            extend_antichains = [extend_by_one(starting_antichains[n])]
        break

    # print(extend_antichains)
    for k in range(n+1, starting_antichains.shape[0]):
        extend_antichains += [extend_by_one(starting_antichains[k])]
        if k % 25000 == 0:
            print(k)

    return extend_antichains


def clean(arr):
    arr.sort(axis=1)
    arr = np.unique(arr, axis=0)
    return arr


compared = np.array([[compare(all_vectors[r], all_vectors[s])
                      for s in range(len(all_vectors))] for r in range(len(all_vectors))], dtype=np.bool_)


path0 = Path(
    '~\Documents\PythonProjects\Antichain Data\length_three').expanduser()
path0.mkdir(parents=True, exist_ok=True)

path1 = Path(
    '~\Documents\PythonProjects\Antichain Data\length_four').expanduser()
path1.mkdir(parents=True, exist_ok=True)


# antichains = np.load(path0/'all length three.npy')

# print(antichains.shape)

# n = antichains.shape[0] // 250000
# print(n)
# for k in range(n):
#     if antichains.shape[0] >= k * 250000:
#         np.save(path0/f'Length three {k}',
#                 antichains[k*250000:(k+1)*250000])
#     else:
#         np.save(path0/f'Length three {k}', antichains[k*250000:])

# antichains = np.load(path0/'Length three 1.npy')
# print(antichains.shape)


# for k in range(1, 230):
#     print(f'Loop number {k}')
#     antichains = np.load(path0/f'Length three {k}.npy')
#     # print()
#     start_time = time.perf_counter()
#     ext_array = extend_array_1(antichains)
#     end_time = time.perf_counter()
#     print(f"Extened time : {end_time-start_time:0.6f}")
#     print(len(ext_array))
#     start_time = time.perf_counter()
#     ar = np.array(ext_array, dtype='int16')
#     end_time = time.perf_counter()
#     print(ar.shape)
#     del ext_array
#     print(f"Create Array : {end_time-start_time:0.6f}")
#     start_time = time.perf_counter()
#     ar = clean(ar)
#     end_time = time.perf_counter()
#     print(f"Clean array time : {end_time-start_time:0.6f}")
#     print(ar.shape)

#     np.save(path1/f'Length four {k}', ar)

#     del ar
# print()

@njit()
def delete_duplicate_rows(matrix_1, matrix_2):
    bool_mask = np.ones(matrix_1.shape[0], dtype='bool')
    # print(bool_mask)
    matrix_1_index = 0
    matrix_2_index = 0
    matrix_1_len = matrix_1.shape[0] - 1
    while matrix_2_index < matrix_2.shape[0]:
        find = True
        for k in range(matrix_2.shape[1]):
            if matrix_2[matrix_2_index, k] < matrix_1[matrix_1_index, k]:
                matrix_2_index += 1
                find = False
                break
            elif matrix_2[matrix_2_index, k] > matrix_1[matrix_1_index, k]:
                matrix_1_index += 1
                find = False
                break
        if find:
            bool_mask[matrix_1_index] = False
            matrix_1_index += 1
        if matrix_1_index > matrix_1_len:
            break
    return matrix_1[bool_mask]


# antichains_1 = np.load(path1/f'Length four 1.npy')
# antichains_2 = np.load(path1/f'Length four 2.npy')

# print(antichains_2.shape[0])

# start_time = time.perf_counter()
# delete_duplicate_rows(antichains_2[:1000000], antichains_1[:10000])

# end_time = time.perf_counter()
# print(f"Extened time : {end_time-start_time:0.6f}")

# start_time = time.perf_counter()
# print(delete_duplicate_rows(antichains_2,
#       antichains_1).shape[0] + antichains_1.shape[0])
# end_time = time.perf_counter()

# print(f"Extened time : {end_time-start_time:0.6f}")

# start_time = time.perf_counter()
# print(clean(np.append(antichains_1,
#       antichains_2, axis=0)).shape[0])
# end_time = time.perf_counter()

# print(f"Extened time : {end_time-start_time:0.6f}")

# for n in range(229):
#     antichains_1 = np.load(path1/f'Length four {n}.npy')
#     for k in range(n+1, 230):
#         antichains_2 = np.load(path1/f'Length four {k}.npy')
#         np.save(
#             path1/f'Length four {k}', delete_duplicate_rows(antichains_2, antichains_1))
#         print(f'n : {n}, k : {k}')


# print()
# start_time = time.perf_counter()
# ext_array = extend_array_1(antichains[:250000])
# end_time = time.perf_counter()
# print(f"Extened time : {end_time-start_time:0.6f}")
# print(len(ext_array))
# start_time = time.perf_counter()
# ar = np.array(ext_array, dtype='int16')
# end_time = time.perf_counter()
# print(ar.shape)
# del ext_array
# print(f"Create Array : {end_time-start_time:0.6f}")
# start_time = time.perf_counter()
# ar = clean(ar)
# end_time = time.perf_counter()
# print(f"Clean array time : {end_time-start_time:0.6f}")
# print(ar.shape)

# np.save(path1/f'Length four {k}', ar)
antichains_1 = np.load(path1/f'Length four 1.npy')
# antichains_2 = np.load(path1/f'Length four 2.npy')


@njit()
def build_volume_vectors(anti_chains):
    volume_vector = np.zeros(
        (anti_chains.shape[0], 9*3*anti_chains.shape[1], 9*3*anti_chains.shape[1]*2), dtype='int8')
    subsets = binary_vectors(anti_chains.shape[1])
    union_size = np.array([[np.sum(np.logical_or(subset_1, subset_2))
                            for subset_1 in subsets] for subset_2 in subsets], dtype='int8')
    subset_size = np.array([np.sum(subset)
                           for subset in subsets], dtype='int8')
    # print(getsizeof(volume_vector) / (8*10**9))
    for n in range(anti_chains.shape[0]):
        vectors = all_vectors[anti_chains[n]]
        # print(vectors)
        for k in range(1, subsets.shape[0]):
            for j in range(1, subsets.shape[0]):
                # subset_1 = subsets[k]
                # subset_2 = subsets[j]
                # union_size = np.sum(np.logical_or(subsets[k], subsets[j]))
                # subset_1_size = np.sum(subsets[k])
                # subset_2_size = np.sum(subsets[j])
                sum_vectors_subset_1 = np.sum(vectors[subsets[k]])
                sum_vectors_subset_2 = np.sum(vectors[subsets[j]])
                volume_vector[n, union_size[k, j], sum_vectors_subset_1 +
                              sum_vectors_subset_2] += (-1)**(subset_size[k]+subset_size[j])

                # volume_vector[n, np.sum(np.logical_or(subsets[k], subsets[j])), np.sum(vectors[subsets[k]]) +
                #               np.sum(vectors[subsets[j]])] += (-1)**(np.sum(subsets[k])+np.sum(subsets[j]))
    return volume_vector


@njit()
def dependence(anti_chains):
    volume_vector = np.zeros(
        (anti_chains.shape[0], 9*3*anti_chains.shape[1], 9*3*anti_chains.shape[1]*2), dtype='int8')
    subsets = binary_vectors(anti_chains.shape[1])
    union_size = np.array([[np.sum(np.logical_or(subset_1, subset_2))
                            for subset_1 in subsets] for subset_2 in subsets], dtype='int8')
    subset_size = np.array([np.sum(subset)
                           for subset in subsets], dtype='int8')
    m = 30
    value_vector = np.array([(3**k*max(3**(m-k)-1, 0))/(3**m-1) for k in range(
        9*3*anti_chains.shape[1])], dtype='float')
    # print(getsizeof(volume_vector) / (8*10**9))
    for n in range(anti_chains.shape[0]):
        vectors = all_vectors[anti_chains[n]]
        # sum = np.zeros(9*3*anti_chains.shape[1], dtype='float')
        sum = 0
        for k in range(1, subsets.shape[0]):
            for j in range(1, subsets.shape[0]):
                sum_vectors_subset_1 = np.sum(vectors[subsets[k]])
                sum_vectors_subset_2 = np.sum(vectors[subsets[j]])
                multiplier = (-1)**(subset_size[k]+subset_size[j])*2.**(-(sum_vectors_subset_1 +
                                                                          sum_vectors_subset_2))
                # for m in range(9*3*anti_chains.shape[1]-1):
                #     sum[m] += multiplier * value_vector[m, union_size[k, j]]
                sum += multiplier * value_vector[union_size[k, j]]

    return 0


n = 20000
# set_num_threads(16)
# start_time = time.perf_counter()
build_volume_vectors(antichains_1[:100])
# print(getsizeof(antichains_1[:1000000]) / (8*10**9))
# end_time = time.perf_counter()
# print(f"Extened time : {end_time-start_time:0.6f}")

start_time = time.perf_counter()
print(getsizeof(build_volume_vectors(antichains_1[:n])) / (8*10**9))
# print(getsizeof(antichains_1[:1000000]) / (8*10**9))
end_time = time.perf_counter()
print(f"Buid time : {end_time-start_time:0.6f}")

print()
# start_time = time.perf_counter()
build_volume_vectors(antichains_1[:100])
# # print(getsizeof(antichains_1[:1000000]) / (8*10**9))
# end_time = time.perf_counter()
# print(f"Extened time : {end_time-start_time:0.6f}")

start_time = time.perf_counter()
dependence(antichains_1[:n])
# print(getsizeof(antichains_1[:1000000]) / (8*10**9))
end_time = time.perf_counter()
print(f"Dependence time : {end_time-start_time:0.6f}")
# print(getsizeof(np.zeros((antichains_1.shape[0]), dtype='float')) / (8*10**9))

# subsets = binary_vectors(3)
# print(np.array([[np.sum(np.logical_or(subset_1, subset_2))
#                  for subset_1 in subsets] for subset_2 in subsets], dtype='int8'))
# print(np.array([np.sum(subset)
#                 for subset in subsets], dtype='int8'))
