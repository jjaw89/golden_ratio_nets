from cmath import sqrt
import numpy as np
import time
import sys
import cupy as cp
import math
import scipy.special

# arr = np.random.randn(3, 3)
# print(arr)
# print(1-arr[:, 0])
# print(np.multiply(arr[:, 2], 1-arr[:, 0]))

# for index in range(0, 5):
#     print(index)

# a = np.arange(12).reshape(6, 2)
# print(a)
# print(np.delete(a, 1, 1))

# d = 2

# a = np.arange(24).reshape(8, 3)
# b = a

# n = a.shape[0]
# m = b.shape[0]

# mu = np.median(a[:, d-1])

# bool_arr = a[:, d-1] <= mu

# a_l = a[bool_arr]
# a_r = a[np.logical_not(bool_arr)]

# bool_arr = b[:, d-1] <= mu

# b_l = b[bool_arr]
# b_r = b[np.logical_not(bool_arr)]

# a_l_tilde = a_l
# a_r_bar = a_r

# b_l_tilde = b_l
# b_r_bar = b_r

# a_l_tilde = np.delete(a_l_tilde, d-1, 1)
# a_r_bar = np.delete(a_r_bar, d-1, 1)
# b_l_tilde = np.delete(b_l_tilde, d-1, 1)
# b_r_bar = np.delete(b_r_bar, d-1, 1)

# a_l_tilde_last_column = np.array(
#     [a_l[index_j, -1]*a_l[index_j, -2] for index_j in range(a_l.shape[0])])

# print(a_l_tilde_last_column)
# print(a_l[:, -1]*a_l[:, -2])
# b_l_tilde_last_column = np.array(
#     [b_l[index_j, -1]*b_l[index_j, -2] for index_j in range(b_l.shape[0])])

# a_l_tilde[:, -1] = a_l_tilde_last_column
# b_l_tilde[:, -1] = b_l_tilde_last_column

# print(f"mu : {mu}")

# print(f"a : {a}")
# print(f"b : {b}")

# print(f"a_l : {a_l}")
# print(f"b_l : {b_l}")

# print(f"a_r : {a_r}")
# print(f"b_r : {b_r}")

# print(f"a_l_tilde : {a_l_tilde}")
# print(f"b_l_tilde : {b_l_tilde}")

# print(f"a_r_bar : {a_r_bar}")
# print(f"b_r_bar : {b_r_bar}")

print(2./3.)
