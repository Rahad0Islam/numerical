import numpy as np

# Function for LU Decomposition (without pivoting)
def LU_Decomposition(A):
    A = A.astype(float)   # Ensure matrix uses floating point numbers
    n = A.shape[0]        # Number of rows (size of matrix)
    L = np.zeros((n, n))  # Initialize L (lower triangular matrix) with zeros
    U = A.copy()          # Copy of A that will be transformed into U

    # Set diagonal of L to 1 (as per LU decomposition convention)
    for i in range(n):
        L[i, i] = 1.0

    # Forward elimination process to build L and U
    for i in range(n - 1):
        if U[i, i] == 0:   # Check for zero pivot (no pivoting implemented here)
            print("Can't apply forward elimination: zero pivot.")
            return None, None
        for j in range(i + 1, n):
            # Compute elimination factor
            factor = U[j, i] / U[i, i]
            L[j, i] = factor  # Store factor in L
            # Subtract factor * pivot row from current row to eliminate entry
            U[j, i:] -= factor * U[i, i:]

    return L, U


# Function to compute determinant from U
def determinant(U):
    det = 1
    # Determinant = product of diagonal elements of U
    for i in range(len(U)):
        det *= U[i][i]
    return det


# Main program
n = int(input("Enter the size of the matrix (n): "))

print("Enter the coefficient matrix row by row (space separated):")
A = []
for i in range(n):
    # Read each row of the matrix from user
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)

# Convert input list to numpy array
A = np.array(A)

# Perform LU Decomposition
L, U = LU_Decomposition(A)

if L is not None and U is not None:
    # Print L
    print("\nLower triangular matrix (L):")
    for row in L:
        print(" ".join(f"{val:.6f}" for val in row))

    # Print U
    print("\nUpper triangular matrix (U):")
    for row in U:
        print(" ".join(f"{val:.6f}" for val in row))

    # Compute determinant
    det = determinant(U)
    print(f"\nDeterminant of the matrix: {det:.6f}")
