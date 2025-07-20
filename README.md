# linear_algebra

A simple Python library for linear algebra operations, built on top of NumPy.

## Features

## Installation
```
pip install .
```

## Usage
```python
from linear_algebra import dot, cross, norm
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(dot(v1, v2))
print(cross(v1, v2))
print(norm(v1))
```
# Linear Algebra Python Library

This is a simple Python library for basic linear algebra operations implemented from scratch (without NumPy).

## Features
- Matrix addition
- Dot product (for vectors)
- Cross product (for 3D vectors)
- Vector norm (magnitude)
- Matrix transpose

More features will be added soon!

## Example Usage
```python
from linear_algebra.operations import add, dot, cross, norm, transpose

a = [[1, 2, 3], [4, 5, 6]]
b = [[4, 5, 6], [7, 8, 9]]
print("Addition:", add(a, b))

x = [1, 2, 3]
y = [4, 5, 6]
print("Dot product:", dot(x, y))
print("Cross product:", cross(x, y))
print("Norm:", norm(x))
print("Transpose:", transpose(a))
```

## How to Use
1. Clone this repository:
   ```
   git clone <repo-url>
   ```
2. Import and use the functions in your Python code as shown above.

## License
MIT
