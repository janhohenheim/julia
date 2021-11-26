
import numba
from numba import jit, vectorize
import numpy as np

@vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def _abs_square(z: complex) -> int:
    return z.real ** 2 + z.imag ** 2


@jit(nopython=True, parallel=True)
def get_julia_set(
    width: int, height: int, c: complex, r: int, max_iter: int
) -> np.ndarray:
    julia_set = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            zx = np.interp(i, (0, width), (-r, r))
            zy = np.interp(j, (0, height), (-r, r))
            z = complex(zx, zy)
            for iter_ in range(max_iter):
                if _abs_square(z) > r ** 2:
                    julia_set[j, i] = iter_
                    break
                z = z ** 2 + c
    return julia_set
    