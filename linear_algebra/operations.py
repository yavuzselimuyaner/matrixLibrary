import random
from sympy import *
"""
This module provides linear algebra operations.
Write your own algorithms below.

Example matrices for testing:
a = [
    [1, 2, 3],
    [4, 5, 6]
]
b = [
    [7, 8, 9],
    [10, 11, 12]
]
"""

# Example function templates:
def add(a,b):
    if(len(a) != len(b) or len(a[0]) != len(b[0])):
        raise ValueError("Matrices must have the same dimensions for addition.")
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
def subtract(a, b):
    if(len(a) != len(b) or len(a[0]) != len(b[0])):
        raise ValueError("Matrices must have the same dimensions for subtraction.")
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
def dot(a, b):
    """Dot product of two arrays."""
    if len(a) != len(b):
        raise ValueError("Inner dimensions must match for dot product.")

    
    return [a[i]*b[i] for i in range(len(a))]
def multiply(a, b):
    """Matrix or scalar multiplication."""
    # Scalar multiplication: if b is int or float
    if isinstance(b, (int, float)):
        return [[a[i][j] * b for j in range(len(a[0]))] for i in range(len(a))]
    # Scalar multiplication: if a is int or float
    if isinstance(a, (int, float)):
        return [[a * b[i][j] for j in range(len(b[0]))] for i in range(len(b))]
    # Matrix multiplication
    if len(a[0]) != len(b):
        raise ValueError("Inner dimensions must match for matrix multiplication.")
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]

def cross(a, b):
    """Cross product of two arrays."""
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Cross product is only defined for 3-dimensional vectors.")
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]

def norm(a):
    """Norm (magnitude) of a vector."""
    return sum(x**2 for x in a) ** 0.5

def transpose(matrix):
    """Transpose a matrix."""
    if not matrix:
        return []
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
def determinant(matrix):
    if(len(matrix) != len(matrix[0])):
        raise ValueError("Matrix must be square for determinant.")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for c in range(len(matrix)):
        submatrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(submatrix)
    return det

def adjoint(matrix):
    """Adjoint of a matrix."""
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square for adjoint.")
    cofactors = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix)):
            submatrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            cofactor_row.append(((-1) ** (i + j)) * determinant(submatrix))
        cofactors.append(cofactor_row)
    return transpose(cofactors)

def inverse(matrix):
    """Inverse of a matrix."""
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    adj = adjoint(matrix)
    return [[adj[i][j] / det for j in range(len(adj[0]))] for i in range(len(adj))]


def random_matrix(rows, cols):
        """Generate a random matrix of given dimensions without numpy."""
        return [[int(random.random()*10) for _ in range(cols)] for _ in range(rows)]

def identity_matrix(size):
    """Generate a random identity matrix of given size."""
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

def eigenvalues (matrix):
    """Calculate eigenvalues of a 2x2 matrix manually (characteristic polynomial)."""
    if len(matrix) != 2 or len(matrix[0]) != 2:
        raise ValueError("This manual eigenvalue function only supports 2x2 matrices.")
    a, b = matrix[0]
    c, d = matrix[1]
    # Characteristic equation: (a - λ)(d - λ) - b*c = 0
    # λ^2 - (a + d)λ + (ad - bc) = 0
    trace = a + d
    det = a * d - b * c
    # Solve quadratic equation: λ^2 - trace*λ + det = 0
    discriminant = trace**2 - 4*det
    if discriminant < 0:
        # Complex eigenvalues
        real = trace / 2
        imag = (abs(discriminant) ** 0.5) / 2
        return [complex(real, imag), complex(real, -imag)]
    else:
        sqrt_disc = discriminant ** 0.5
        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2
        return [lambda1, lambda2]
def eigenvectors(matrix):
    """Calculate eigenvectors of a 2x2 matrix for each eigenvalue."""
    eigens = eigenvalues(matrix)
    vectors = []
    for eig in eigens:
        # (A - λI) * [x, y]^T = 0
        A = [[matrix[i][j] - (eig if i == j else 0) for j in range(2)] for i in range(2)]
        a, b = A[0]
        c, d = A[1]
        # Çözüm: x veya y serbest, diğeri ona bağlı
        if abs(b) > 1e-12:
            x = 1
            y = -a / b
        elif abs(a) > 1e-12:
            x = -b / a
            y = 1
        elif abs(d) > 1e-12:
            x = 1
            y = -c / d
        else:
            x = 1
            y = 0
        vectors.append([x, y])
    return vectors
