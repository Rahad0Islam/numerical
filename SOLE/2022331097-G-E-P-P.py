import numpy as np

# Function to calculate dot product between two vectors
def sum_clc(a, b):
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

# Gaussian Elimination with Partial Pivoting
def Partial_Pivoting(A, b):
    A = A.astype(float)            # Convert matrix A to float type
    b = b.astype(float).flatten()  # Convert RHS vector b to 1D float array
    n = len(b)                     # Number of equations (size of system)

    # Forward Elimination with Partial Pivoting
    for i in range(n-1):
        # Step 1: Find row with maximum absolute value in column i (pivot row)
        max_row = np.argmax(abs(A[i:, i])) + i
        
        # If pivot element is zero even after row swapping -> no unique solution
        if A[max_row, i] == 0:
            print("Zero pivot element detected. Cannot solve.")
            return
        
        # Step 2: Swap rows in A and corresponding b values
        A[[i, max_row]] = A[[max_row, i]]
        b[i], b[max_row] = b[max_row], b[i]
        
        # Step 3: Eliminate entries below the pivot
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]   # Update row j of A
            b[j] -= factor * b[i]           # Update RHS vector

    # If the last pivot is zero, system cannot be solved
    if A[-1, -1] == 0:
        print("Zero pivot element detected. Cannot solve.")
        return

    # Back Substitution
    x = np.zeros(n)                # Initialize solution vector
    x[-1] = b[-1] / A[-1, -1]      # Solve last variable directly
    for i in range(n-2, -1, -1):   # Loop from second-last row up to first
        if A[i, i] == 0:
            print("Zero pivot element detected during back substitution. Cannot solve.")
            return
        result = sum_clc(A[i, i+1:], x[i+1:])  # Compute sum of known terms
        x[i] = (b[i] - result) / A[i, i]       # Solve for current variable

    return x


# ---------------- Main Program ----------------

n = int(input("Enter the size of the matrix (n): "))

print("Enter the coefficient matrix row by row (space separated):")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)

print("Enter the RHS vector row by row (one value per line):")
b = []
for i in range(n):
    val = float(input(f"b[{i+1}]: "))
    b.append([val])

A = np.array(A)
b = np.array(b)

solution = Partial_Pivoting(A, b)

# Print solution if found
if solution is not None:
    print("\nSolution vector (x):")
    for i, val in enumerate(solution, start=1):
        print(f"x{i} = {val}")
