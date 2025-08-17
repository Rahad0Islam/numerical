import numpy as np

# Function to perform LU Decomposition of matrix A
def LU_Decomposition(A):
    A = A.astype(float)         # Convert entries to float (for safe division)
    n = A.shape[0]              # Number of rows (and columns since A is square)
    L = np.zeros((n, n))        # Initialize L (lower triangular) as zero matrix
    U = A.copy()                # Copy A to U (will be transformed to upper triangular)

    # Step 1: Set diagonal elements of L to 1
    for i in range(n):
        L[i, i] = 1.0

    # Step 2: Perform Gaussian elimination to transform A into U
    for i in range(n - 1):
        if U[i, i] == 0:        # If pivot is zero, cannot proceed
            print("Can't apply forward elimination: zero pivot.")
            return None, None
        for j in range(i + 1, n):               # Eliminate entries below pivot
            factor = U[j, i] / U[i, i]          # Compute multiplier
            L[j, i] = factor                    # Store multiplier in L
            U[j, i:] -= factor * U[i, i:]       # Update row j in U (row operation)

    return L, U


# ----------- MAIN PROGRAM -----------

# Input matrix size
n = int(input("Enter the size of the matrix (n): "))

print("Enter the coefficient matrix row by row (space separated):")
A = []
for i in range(n):
    # Read each row as float values
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)

A = np.array(A)    # Convert list of lists into numpy array

# Perform LU decomposition
L, U = LU_Decomposition(A)

# Print results if decomposition was successful
if L is not None and U is not None:
    print("\nLower triangular matrix (L):")
    for row in L:
        print(" ".join(f"{val:.6f}" for val in row))

    print("\nUpper triangular matrix (U):")
    for row in U:
        print(" ".join(f"{val:.6f}" for val in row))
