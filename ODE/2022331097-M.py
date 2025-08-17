import matplotlib.pyplot as plt
import sympy as sp
import math

# -----------------------------------------------------
# Function: Midpoint Method Implementation
# -----------------------------------------------------
def midpoint_method(f, x_initial, y_initial, step_size, num_steps):
    """
    Implements the Midpoint Method (second-order Runge-Kutta) for solving ODEs.

    Parameters:
    f          -> function representing dy/dx = f(x, y)
    x_initial  -> initial x value
    y_initial  -> initial y value
    step_size  -> step size (h)
    num_steps  -> number of steps

    Returns:
    x_values -> list of x values
    y_values -> list of approximate y values
    """
    x_values, y_values = [x_initial], [y_initial]

    x_current = x_initial
    y_current = y_initial

    for _ in range(num_steps):
        # Step 1: Calculate slope at beginning of interval
        k1 = f(x_current, y_current)
        
        # Step 2: Estimate slope at midpoint
        k2 = f(x_current + step_size / 2, y_current + step_size * k1 / 2)
        
        # Step 3: Update y using slope at midpoint
        y_current += step_size * k2
        # Step 4: Update x
        x_current += step_size

        # Store current values
        x_values.append(x_current)
        y_values.append(y_current)

    return x_values, y_values

# -----------------------------------------------------
# Step 1: Input function safely using SymPy
# -----------------------------------------------------
print("Use only polynomials in x and y (e.g., x + y, x**2 - 3*y) :")
func_str = input("Enter dy/dx as a polynomial in x and y: ")
func_str = func_str.replace("^", "**")

# Define symbolic variables
x, y = sp.symbols("x y")

try:
    sym_expr = sp.sympify(func_str)
except sp.SympifyError:
    print("Invalid expression. Please enter a valid polynomial in x and y.")
    exit()

# Convert symbolic expression to Python function
f = sp.lambdify((x, y), sym_expr, modules=["math"])

# -----------------------------------------------------
# Step 2: Initial conditions & parameters
# -----------------------------------------------------
x_initial = float(input("Enter initial x (x0): "))
y_initial = float(input("Enter initial y (y0): "))
step_size = float(input("Enter step size (h): "))
num_steps = int(input("Enter number of steps (n): "))

# -----------------------------------------------------
# Step 3: Run Midpoint Method
# -----------------------------------------------------
x_values, y_values = midpoint_method(f, x_initial, y_initial, step_size, num_steps)

# -----------------------------------------------------
# Step 4: Print results
# -----------------------------------------------------
print("\nMidpoint Method Results:\n")
print("Step\t   X\t\t   Y")
print("-" * 34)
for step in range(len(x_values)):
    print(f"{step}\t {x_values[step]:.6f}\t {y_values[step]:.6f}")

# -----------------------------------------------------
# Step 5: Plot Graph
# -----------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, "gs-", label="Midpoint Approximation")  # green line with square markers
plt.xlabel("x")
plt.ylabel("y")
plt.title("Midpoint Method Solution")
plt.legend()
plt.grid(True)
plt.show()
