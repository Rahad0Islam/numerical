import numpy as np

# Function to calculate dot product of two vectors (used in back substitution)
def sum_clc(a, b):
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]   # multiply corresponding elements and add
    return total


# Function to solve system of linear equations Ax = b using Gaussian Elimination
def GaussElimination(A, b):
    A = A.astype(float)              # convert matrix A to float type
    b = b.astype(float).flatten()    # convert b to 1D float array
    n = len(b)                       # number of equations / unknowns

    # ---- Forward Elimination ----
    for i in range(n-1):             # loop over pivot rows
        if A[i, i] == 0:             # check pivot element
            print("Can't apply Gaussian elimination: zero pivot")
            return
        for j in range(i+1, n):      # eliminate values below the pivot
            factor = A[j, i] / A[i, i]        # multiplier for row operation
            A[j, i:] = A[j, i:] - factor * A[i, i:]   # update row j of A
            b[j] = b[j] - factor * b[i]               # update RHS vector b

    # ---- Back Substitution ----
    x = np.zeros(n)                  # initialize solution vector x
    x[n-1] = b[n-1] / A[n-1, n-1]    # solve last variable directly
    for i in range(n-2, -1, -1):     # move upward through rows
        result = sum_clc(A[i, i+1:], x[i+1:])         # compute sum(A[i][k]*x[k])
        x[i] = (b[i] - result) / A[i, i]              # solve for x[i]

    return x


# ---- Main Program ----
n = int(input("Enter the size of the matrix (n): "))

print("Enter the coefficient matrix row by row (space separated):")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))  # read row of A
    A.append(row)

print("Enter the RHS vector row by row (one value per line):")
b = []
for i in range(n):
    val = float(input(f"b[{i+1}]: "))   # read each value of RHS vector b
    b.append([val])

A = np.array(A)   # convert list into numpy array
b = np.array(b)

# Solve system
solution = GaussElimination(A, b)

# Print solution
print("\nSolution vector (x):")
for i, val in enumerate(solution, start=1):
    print(f"x{i} = {val}")
