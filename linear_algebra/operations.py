
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
def dot(a, b):
    """Dot product of two arrays."""
    if len(a) != len(b):
        raise ValueError("Inner dimensions must match for dot product.")
    
    return [a[i]*b[i] for i in range(len(a))]

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

a = [[1,2,3], [4,5,6]]
b = [[4,5,6], [7,8,9]]
print("Addition:", add(a, b))

x = [1, 2, 3]
y = [4, 5, 6]
print("Dot product:", dot(x, y))
print("transpose:", transpose(a))