import matplotlib.pyplot as plt
import math
import sympy as sp

# -----------------------------------------------------
# Function: Euler Method Implementation
# -----------------------------------------------------
def euler_method(f, x_start, y_start, step_size, steps):
    """
    Implements Euler's Method for solving ODEs.

    Parameters:
    f          -> function representing dy/dx = f(x, y)
    x_start    -> initial x value
    y_start    -> initial y value
    step_size  -> step size (h)
    steps      -> number of steps

    Returns:
    xs -> list of x values
    ys -> list of approximate y values
    """
    xs, ys = [x_start], [y_start]

    for _ in range(steps):
        # Euler formula: y_{n+1} = y_n + h*f(x_n, y_n)
        y_start += step_size * f(x_start, y_start)
        x_start += step_size

        xs.append(x_start)
        ys.append(y_start)

    return xs, ys


# -----------------------------------------------------
# Step 1: Take ODE input safely with SymPy
# -----------------------------------------------------
ode_expr = input("Enter dy/dx as a function of x and y (e.g., x + y, x^2 - 3*y): ")
ode_expr = ode_expr.replace("^", "**")  # allow ^ for power

# Define symbolic variables
x, y = sp.symbols("x y")

# Parse expression safely
try:
    sym_expr = sp.sympify(ode_expr)
except sp.SympifyError:
    print("Invalid expression. Please enter a valid function in x and y.")
    exit()

# Convert to Python function
f = sp.lambdify((x, y), sym_expr, modules=["math"])

# -----------------------------------------------------
# Step 2: Initial conditions & parameters
# -----------------------------------------------------
x_initial = float(input("Enter the initial x value: "))
y_initial = float(input("Enter the initial y value: "))
h = float(input("Enter the step size h: "))
num_steps = int(input("Enter the number of steps: "))

# -----------------------------------------------------
# Step 3: Run Euler's Method
# -----------------------------------------------------
x_vals, y_vals = euler_method(f, x_initial, y_initial, h, num_steps)

# -----------------------------------------------------
# Step 4: Print results in tabular form
# -----------------------------------------------------
print("\nEuler's Method Results:\n")
print("step\t   X\t\t   Y (Euler)")
print("-" * 34)
for i in range(len(x_vals)):
    print(f"{i}\t {x_vals[i]:.6f}\t {y_vals[i]:.6f}")

# -----------------------------------------------------
# Step 5: Plot Graph
# -----------------------------------------------------
plt.figure(figsize=(18, 16))
plt.plot(x_vals, y_vals, "ro-", label="Euler Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Euler's Method Solution")
plt.legend()
plt.grid(True)
plt.show()
