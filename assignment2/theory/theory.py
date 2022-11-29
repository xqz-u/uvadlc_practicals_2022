import numpy as np
from numpy import linalg as L

A3 = np.array(
    [
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ]
)
A3_squared = A3 @ A3

assert (np.matmul(A3, A3) == A3_squared).all()
assert (A3_squared == L.matrix_power(A3, 2)).all()

L.matrix_power(A3, 3)
