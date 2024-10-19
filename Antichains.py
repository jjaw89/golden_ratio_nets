
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
from numba import jit, njit, vectorize, prange
from sys import getsizeof

# n = 10
# print(len([[x, y, 0] for x in range(0, n) for y in range(0, x)]))

n = 10
all_vectors = np.array([[x, y, z]
                        for x in range(0, n) for y in range(0, n) for z in range(0, n)], dtype='int16')
num_vectors = all_vectors.shape[0]
numbers = np.arange(1000, dtype='int16')


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


start_time = time.perf_counter()
compared = np.array([[compare(all_vectors[r], all_vectors[s])
                      for s in range(len(all_vectors))] for r in range(len(all_vectors))], dtype=np.bool_)
end_time = time.perf_counter()


antichains = np.vstack(np.arange(num_vectors, dtype=np.int16))


# print(compared)
# print(antichains)

# compared_row_index = 0
# print(antichains[1])
# print(compared[2, 3])


starting_row = 1
vect = np.array([k for k in range(num_vectors) if compared[starting_row, k]])
vect = np.vstack(vect)

duplicated_row = np.array([antichains[starting_row]]*np.shape(vect)[0])
extended_antichains = np.append(duplicated_row, vect, axis=1)

for row_index in range(starting_row+1, num_vectors):
    vect = np.array([k for k in range(num_vectors) if compared[row_index, k]])
    if np.size(vect) != 0:
        vect = np.vstack(vect)
        duplicated_row = np.array([antichains[row_index]]*np.shape(vect)[0])
        added_antichains = np.append(duplicated_row, vect, axis=1)
        extended_antichains = np.append(
            extended_antichains, added_antichains, axis=0)

extended_antichains = np.sort(extended_antichains, axis=1)
# print(extended_antichains)
extended_antichains = np.unique(extended_antichains, axis=0)
# print(extended_antichains)

print(f"Number of vectors : {num_vectors}")
print(f"Number of subsets of size 2 : {num_vectors * (num_vectors-1)}")
print(f"Shape of antichains : {np.shape(extended_antichains)}")
print(
    f"Percent that are antichains : {np.shape(extended_antichains)[0] / (num_vectors * (num_vectors-1))*100}")
print(f"Total time : {end_time-start_time:0.6f}")
process = psutil.Process(os.getpid())
print(process.memory_info().rss / 1000000)  # in megabytes


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


# print()
# start_time = time.perf_counter()
# ext_array = extend_array_1(extended_antichains[:10000])
# end_time = time.perf_counter()
# print(f"Total time : {end_time-start_time:0.6f}")
# print(len(ext_array))
# start_time = time.perf_counter()
# ar = np.array(ext_array, dtype='int16')
# print(getsizeof(ext_array) // 8000)
# del ext_array
# print(getsizeof(ar) // 8000)
# end_time = time.perf_counter()
# print(f"Total time : {end_time-start_time:0.6f}")
# start_time = time.perf_counter()
# clean(ar)
# end_time = time.perf_counter()


# print(f"Total time : {end_time-start_time:0.6f}")
# print(ar.shape)
# print(getsizeof(ar) // 8000)
# print()


print()
start_time = time.perf_counter()
ext_array = extend_array_1(extended_antichains)
end_time = time.perf_counter()
print(f"Total time : {end_time-start_time:0.6f}")
print(len(ext_array))
print(getsizeof(ext_array) // 8000)
start_time = time.perf_counter()
ar = np.array(ext_array, dtype='int16')
end_time = time.perf_counter()
print(ar.shape)
print(getsizeof(ar) // 8000)
del ext_array
print(f"Create Array : {end_time-start_time:0.6f}")
start_time = time.perf_counter()
ar = clean(ar)
end_time = time.perf_counter()
print(f"Sort array time : {end_time-start_time:0.6f}")
print(ar.shape)
print(getsizeof(ar) // 8000)
print()


# for n in range(9):
print(n+3)
start_time = time.perf_counter()
if ar.shape[0] > 100000:
    ext_array = extend_array_1(ar[:500000])
else:
    ext_array = extend_array_1(ar)
end_time = time.perf_counter()
print(f"Total time : {end_time-start_time:0.6f}")
print(len(ext_array))
print(getsizeof(ext_array) // 8000)
start_time = time.perf_counter()
ar = np.array(ext_array, dtype='int16')
end_time = time.perf_counter()
print(ar.shape)
print(getsizeof(ar) // 8000)
del ext_array
print(f"Create Array : {end_time-start_time:0.6f}")
start_time = time.perf_counter()
ar = clean(ar)
print(getsizeof(ar) // 8000)
end_time = time.perf_counter()
print(f"Sort array time : {end_time-start_time:0.6f}")
print(ar.shape)
print(getsizeof(ar) // 8000)
print()


path = Path('~\Documents\PythonProjects\Antichain Data\length_three').expanduser()
path.mkdir(parents=True, exist_ok=True)


np.save(path/'all length three', ar)

# start_time = time.perf_counter()
# for _ in range(1):

#     antichains = extended_antichains

#     for starting_index in range(antichains.shape[0]):

#         vectors_that_extend = compared[antichains[starting_index, 0]]
#         for n in range(1, antichains.shape[1]):
#             vectors_that_extend = np.multiply(
#                 vectors_that_extend, compared[antichains[starting_index, n]])

#         if np.count_nonzero(vectors_that_extend == True) != 0:

#             vect = np.array([k for k in range(num_vectors)
#                             if vectors_that_extend[k]])
#             vect = np.vstack(vect)
#             duplicated_row = np.array([antichains[starting_index]]
#                                       * np.shape(vect)[0])
#             extended_antichains = np.append(duplicated_row, vect, axis=1)
#             print(starting_index)
#             break

#     previous_number = extended_antichains.shape[0]
#     for row_index in range(starting_index+1, antichains.shape[0]):
#         vectors_that_extend = compared[antichains[row_index, 0]]
#         for n in range(1, antichains.shape[1]):
#             vectors_that_extend = np.multiply(
#                 vectors_that_extend, compared[antichains[row_index, n]])

#         vect = np.array([k for k in range(num_vectors)
#                         if vectors_that_extend[k]])
#         if np.size(vect) != 0:
#             vect = np.vstack(vect)
#             duplicated_row = np.array(
#                 [antichains[row_index]]*np.shape(vect)[0])
#             added_antichains = np.append(
#                 duplicated_row, vect, axis=1)
#             extended_antichains = np.append(
#                 extended_antichains, added_antichains, axis=0)

#         if row_index % 5000 == 0:
#             extended_antichains = np.sort(extended_antichains, axis=1)
#             # print(extended_antichains)
#             extended_antichains = np.unique(extended_antichains, axis=0)
#             # print(extended_antichains)
#             end_time = time.perf_counter()
#             print(f"{row_index} Total time : {end_time-start_time:0.6f}")
#             print(extended_antichains.shape)
#             print(f"{extended_antichains.shape[0]-previous_number}")
#             start_time = time.perf_counter()
#             previous_number = extended_antichains.shape[0]

#     process = psutil.Process(os.getpid())
#     print(process.memory_info().rss / 1000000)  # in megabytes
