import time
import numpy as np
from linear_algebra import operations as la

# Matrix and vector for testing
A = [[2, -4], [-1, -1]]
B = [[1, 2], [3, 4]]
x = [1, 2]
y = [3, 4]

np_A = np.array(A)
np_B = np.array(B)
np_x = np.array(x)
np_y = np.array(y)

print("--- Addition ---")
start = time.time()
la_add = la.add(A, B)
end = time.time()
print("Manual:", la_add, "Time:", end - start)

start = time.time()
np_add = (np_A + np_B).tolist()
end = time.time()
print("NumPy:", np_add, "Time:", end - start)

print("\n--- Dot Product ---")
start = time.time()
la_dot = sum(la.dot(x, y))
end = time.time()
print("Manual:", la_dot, "Time:", end - start)

start = time.time()
np_dot = np.dot(np_x, np_y)
end = time.time()
print("NumPy:", np_dot, "Time:", end - start)

print("\n--- Matrix Multiplication ---")
start = time.time()
la_mult = la.multiply(A, B)
end = time.time()
print("Manual:", la_mult, "Time:", end - start)

start = time.time()
np_mult = (np_A @ np_B).tolist()
end = time.time()
print("NumPy:", np_mult, "Time:", end - start)

print("\n--- Eigenvalues ---")
start = time.time()
la_eigvals = la.eigenvalues(A)
end = time.time()
print("Manual:", la_eigvals, "Time:", end - start)

start = time.time()
np_eigvals = np.linalg.eigvals(np_A)
end = time.time()
print("NumPy:", np_eigvals.tolist(), "Time:", end - start)

print("\n--- Eigenvectors ---")
start = time.time()
la_eigvecs = la.eigenvectors(A)
end = time.time()
print("Manual:", la_eigvecs, "Time:", end - start)

start = time.time()
_, np_eigvecs = np.linalg.eig(np_A)
end = time.time()
print("NumPy:", np_eigvecs.tolist(), "Time:", end - start)
import time
import numpy as np
from linear_algebra import operations as la

# 2x2 matrices for most tests
A = [[2, -4], [-1, -1]]
B = [[1, 2], [3, 4]]
# 3D vectors for cross/norm
x = [1, 2, 3]
y = [4, 5, 6]

np_A = np.array(A)
np_B = np.array(B)
np_x = np.array(x)
np_y = np.array(y)

print("--- Subtract ---")
start = time.time()
manual = la.subtract(A, B)
manual_time = time.time() - start
start = time.time()
numpy = (np_A - np_B).tolist()
numpy_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print(f"NumPy: {numpy} (Time: {numpy_time:.8f} s)")

print("\n--- Cross Product ---")
start = time.time()
manual = la.cross(x, y)
manual_time = time.time() - start
start = time.time()
numpy = np.cross(np_x, np_y).tolist()
numpy_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print(f"NumPy: {numpy} (Time: {numpy_time:.8f} s)")

print("\n--- Norm ---")
start = time.time()
manual = la.norm(x)
manual_time = time.time() - start
start = time.time()
numpy = np.linalg.norm(np_x)
numpy_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print(f"NumPy: {numpy} (Time: {numpy_time:.8f} s)")

print("\n--- Transpose ---")
start = time.time()
manual = la.transpose(A)
manual_time = time.time() - start
start = time.time()
numpy = np_A.T.tolist()
numpy_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print(f"NumPy: {numpy} (Time: {numpy_time:.8f} s)")

print("\n--- Determinant ---")
start = time.time()
manual = la.determinant(A)
manual_time = time.time() - start
start = time.time()
numpy = np.linalg.det(np_A)
numpy_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print(f"NumPy: {numpy} (Time: {numpy_time:.8f} s)")

print("\n--- Inverse ---")
start = time.time()
manual = la.inverse(A)
manual_time = time.time() - start
start = time.time()
numpy = np.linalg.inv(np_A).tolist()
numpy_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print(f"NumPy: {numpy} (Time: {numpy_time:.8f} s)")

print("\n--- Random Matrix (3x3) ---")
start = time.time()
manual = la.random_matrix(3, 3)
manual_time = time.time() - start
start = time.time()
numpy = np.random.randint(0, 10, (3, 3)).tolist()
numpy_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print(f"NumPy: {numpy} (Time: {numpy_time:.8f} s)")

print("\n--- Identity Matrix (3x3) ---")
start = time.time()
manual = la.identity_matrix(3)
manual_time = time.time() - start
start = time.time()
numpy = np.eye(3).tolist()
numpy_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print(f"NumPy: {numpy} (Time: {numpy_time:.8f} s)")

print("\n--- Adjoint (no direct NumPy equivalent, only manual) ---")
start = time.time()
manual = la.adjoint(A)
manual_time = time.time() - start
print(f"Manual: {manual} (Time: {manual_time:.8f} s)")
print("NumPy: Not available directly; can be implemented with minors and cofactors.")
