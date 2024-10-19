# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

goldenRatio = (1+np.sqrt(5))/2

endInZero = [[0]]
endInOne = [[1]]
noConsecutiveOnes = [[[0],[1]]]
   

start_time = time.perf_counter()

for i in range(20):
    endInZeroZero = copy.deepcopy(endInZero)
    endInZeroOne = copy.deepcopy(endInZero)
    endInOneZero = copy.deepcopy(endInOne)
    for j in endInZeroZero:
        j.append(0)
    for j in endInZeroOne:
        j.append(1)
    for j in endInOneZero:
        j.append(0)
    endInZero = endInZeroZero + endInOneZero
    endInOne = endInZeroOne
    noConsecutiveOnes.append(endInZero + endInOne)

end_time = time.perf_counter()



print(f"Execution Time : {end_time-start_time:0.6f}")


# =============================================================================
# def add_a_zero(sequences):
#     for sequence in sequences:
#         sequences.append(0)
#     return sequences
# 
# def add_a_one(sequences):
#     for sequence in sequences:
#         sequences.append(1)
#     return sequences
# 
# 
# endInZero = [[0]]
# endInOne = [[1]]
# noConsecutiveOnes = [[[0],[1]]]
# 
# start_time = time.perf_counter()
# 
# for i in range(20):
#     endInZeroZero = add_a_zero(endInZero)
#     endInZeroOne = add_a_one(endInOne)
#     endInOneZero =  add_a_zero(endInOne)
# 
#     endInZero = endInZeroZero + endInOneZero
#     endInOne = endInZeroOne
#     noConsecutiveOnes.append(endInZero + endInOne)
# 
# end_time = time.perf_counter()
# =============================================================================


print(f"Execution Time : {end_time-start_time:0.6f}")
print("finished")



    




