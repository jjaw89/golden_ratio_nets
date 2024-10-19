from numba import jit, njit, vectorize, prange, cuda
import numpy as np
import cupy as cp


@cuda.jit
def test():

    # x_array = np.arange(10)  # type: numpy.ndarray
    # type: numba.cuda.cudadrv.devicearray.DeviceNDArray
    # x_array = cuda.to_device(x)
    x_cupy = cp.array([1], dtype='int16')  # type: cupy.ndarray
    return


test()
