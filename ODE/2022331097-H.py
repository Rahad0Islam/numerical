import matplotlib.pyplot as plt
import sympy as sp
import math

# -----------------------------------------------------
# Function: Heun's Method Implementation
# -----------------------------------------------------
def heun_method(f, x_initial, y_initial, step_size, num_steps):
    """
    Implements Heun's Method (Improved Euler) for solving ODEs.

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
    x_values = [x_initial]  # list to store x values
    y_values = [y_initial]  # list to store y values

    x_current = x_initial
    y_current = y_initial

    # Loop through each step
    for _ in range(num_steps):
        # Step 1: Predictor (Euler step)
        y_predictor = y_current + step_size * f(x_current, y_current)
        
        # Step 2: Corrector (average slope)
        average_slope = (f(x_current, y_current) + f(x_current + step_size, y_predictor)) / 2

        # Step 3: Update y using average slope
        y_current += step_size * average_slope
        # Step 4: Update x
        x_current += step_size

        # Append current values to lists
        x_values.append(x_current)
        y_values.append(y_current)

    return x_values, y_values

# -----------------------------------------------------
# Step 1: Take ODE input safely using SymPy
# -----------------------------------------------------
function_expression = input("Enter dy/dx as a polynomial in x and y (e.g., x + y, x**2 - 3*y) : ")
function_expression = function_expression.replace("^", "**")  # allow ^ for power

# Define symbolic variables
x, y = sp.symbols("x y")

# Parse expression safely using sympy
try:
    sym_expr = sp.sympify(function_expression)
except sp.SympifyError:
    print("Invalid expression. Please enter a valid polynomial in x and y.")
    exit()

# Convert symbolic expression to a callable Python function
f = sp.lambdify((x, y), sym_expr, modules=["math"])

# -----------------------------------------------------
# Step 2: Initial conditions & parameters
# -----------------------------------------------------
x_initial = float(input("Enter initial x (x0): "))
y_initial = float(input("Enter initial y (y0): "))
step_size = float(input("Enter step size (h): "))
num_steps = int(input("Enter number of steps (n): "))

# -----------------------------------------------------
# Step 3: Run Heun's Method
# -----------------------------------------------------
x_values, y_values = heun_method(f, x_initial, y_initial, step_size, num_steps)

# -----------------------------------------------------
# Step 4: Print results in tabular form
# -----------------------------------------------------
print("\nHeun's Method Results:\n")
print("Step\t   X\t\t   Y")
print("-" * 34)
for step in range(len(x_values)):
    print(f"{step}\t {x_values[step]:.6f}\t {y_values[step]:.6f}")

# -----------------------------------------------------
# Step 5: Plot Graph
# -----------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, "bo-", label="Heun Approximation")  # blue line with circle markers
plt.xlabel("x")
plt.ylabel("y")
plt.title("Heun's Method Solution")
plt.legend()
plt.grid(True)
plt.show()
