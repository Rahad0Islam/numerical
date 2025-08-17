import numpy as np

# Function to perform LU Decomposition
def LU_Decomposition(A):
    A = A.astype(float)         # Convert matrix A to float for division operations
    n = A.shape[0]              # Number of rows (and columns, since A is square)
    L = np.zeros((n, n))        # Initialize Lower triangular matrix L with zeros
    U = A.copy()                # Copy of A, will be transformed into Upper triangular matrix U

    # Set diagonal of L to 1 (Doolittle’s method)
    for i in range(n):
        L[i, i] = 1.0

    # Forward elimination process
    for i in range(n - 1):
        if U[i, i] == 0:        # If pivot is zero, LU decomposition fails
            print("Can't apply forward elimination: zero pivot.")
            return None, None
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]   # Multiplier for elimination
            L[j, i] = factor             # Store factor in L
            U[j, i:] -= factor * U[i, i:]  # Eliminate below-diagonal elements in U

    return L, U


# Helper function for dot product (like np.dot but written manually)
def sum_clc(a, b):
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


# Function to calculate the Inverse of a matrix using LU decomposition
def InverseMatrix(A):
    A = A.astype(float)
    n = A.shape[0]
    I = np.eye(n)                      # Identity matrix of size n
    inverse_matrix = np.zeros((n, n))  # To store the final inverse

    # Step 1: Perform LU decomposition
    L, U = LU_Decomposition(A)
    if L is None or U is None:
        return None

    # Step 2: Solve n systems (Ax = e_i, where e_i is ith column of Identity)
    for i in range(n):
        b = I[:, i].copy()   # ith column of Identity (unit vector)

        # Forward substitution to solve Lz = b
        z = np.zeros(n)
        z[0] = b[0] / L[0, 0]
        for j in range(1, n):
            result = sum_clc(L[j, :j], z[:j])
            z[j] = (b[j] - result) / L[j, j]

        # Backward substitution to solve Ux = z
        x = np.zeros(n)
        x[-1] = z[-1] / U[-1, -1]
        for j in range(n - 2, -1, -1):
            result = sum_clc(U[j, j+1:], x[j+1:])
            x[j] = (z[j] - result) / U[j, j]

        # Store the solution vector as ith column of the inverse
        inverse_matrix[:, i] = x

    return inverse_matrix


# ----------- MAIN PROGRAM -----------
n = int(input("Enter the size of the matrix (n): "))

print("Enter the coefficient matrix row by row (space separated):")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)

A = np.array(A)

# Compute inverse
inverse_A = InverseMatrix(A)

# Display result
if inverse_A is not None:
    print("\nInverse Matrix:")
    for row in inverse_A:
        print(" ".join(f"{val:.6f}" for val in row))
