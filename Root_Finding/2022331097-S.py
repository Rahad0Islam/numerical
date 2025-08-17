import math
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Function Parser
# ==============================
# Evaluates the input function string at x.
# Only allows math functions and variable x (safe eval).
def f(x):
    return eval(func_str, {"x": x, "math": math, "__builtins__": None})


# ==============================
# Secant Method Implementation
# ==============================
def secant(x0, x1, tol=1e-6, max_iter=100):
    print(f"\nStarting Secant Method with x0 = {x0}, x1 = {x1}")
    table = []        # To store iteration data
    prev_x2 = None    # For approximate error calculation

    # Iterative process
    for it in range(1, max_iter + 1):
        fx0 = f(x0)
        fx1 = f(x1)

        # Prevent division by zero
        if fx1 - fx0 == 0:
            print("Division by zero. Method fails.")
            return None, []

        # Secant formula: x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = f(x2)

        # --------------------------
        # Approximate Relative Error (%)
        # --------------------------
        if prev_x2 is not None:
            ea = abs((x2 - prev_x2) / x2) * 100
        else:
            ea = None  # First iteration has no error

        # --------------------------
        # Significant Digits (SD) estimation
        # --------------------------
        if ea is not None and ea != 0:
            sd = 0
            threshold = 5 * 10**(-sd)
            while ea < threshold:
                sd += 1
                threshold = 5 * 10**(-sd)
        else:
            sd = "∞" if ea == 0 else "-"  # Infinite SD if exact root found

        # Save iteration info: iteration number, x0, x1, x2, f(x2), error, SD
        table.append([it, x0, x1, x2, fx2, ea, sd])
        prev_x2 = x2

        # --------------------------
        # Stopping Criteria
        # --------------------------
        # Stop if f(x2) is small enough or successive approximations converge
        if abs(fx2) < tol or abs(x2 - x1) < tol:
            break

        # Prepare next iteration: shift x0, x1
        x0, x1 = x1, x2

    # ==============================
    # Print Iteration Table
    # ==============================
    print(f"\n{'Iter':<6} {'x0':>12} {'x1':>12} {'x2':>12} {'f(x2)':>12} {'Error(%)':>12} {'SD':>6}")
    for row in table:
        it, x0_val, x1_val, x2_val, fx2_val, ea, sd = row
        ea_str = f"{ea:>12.6f}" if ea is not None else " " * 12
        print(f"{it:<6} {x0_val:>12.6f} {x1_val:>12.6f} {x2_val:>12.6f} {fx2_val:>12.6f} {ea_str} {str(sd):>6}")

    # Print final root
    print(f"\nRoot found at x = {table[-1][3]:.6f} after {table[-1][0]} iterations")
    return table[-1][3], table


# ==============================
# Input Section
# ==============================
func_str = input("Enter function in terms of x (e.g. x^3 - x - 2): ").replace("^", "**")

try:
    x0 = float(input("Enter first initial guess x0: "))
    x1 = float(input("Enter second initial guess x1: "))
    tol_input = input("Enter tolerance (default 1e-6): ")
    tol = float(tol_input) if tol_input else 1e-6
    iter_input = input("Enter maximum iterations (default 100): ")
    max_iter = int(iter_input) if iter_input else 100
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()


# ==============================
# Run Secant Method
# ==============================
root, table = secant(x0, x1, tol, max_iter)

if root is not None:
    print(f"\nApproximate root: {root:.6f}")

    # ==============================
    # Plotting Section
    # ==============================
    x_vals = np.linspace(root - 5, root + 5, 400)  # Range around root
    y_vals = [f(x) for x in x_vals]

    # Points used in iterations
    x_points = [row[3] for row in table]  # x2 values
    y_points = [f(x) for x in x_points]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {func_str}", color='blue')
    plt.axhline(0, color='black', linewidth=0.5)  # X-axis

    # Plot iteration points and final root
    plt.scatter(x_points, y_points, color='red', label='Approximations', zorder=5)
    plt.scatter([root], [f(root)], color='green', s=100, label='Final Root', edgecolors='black')

    plt.title("Secant Method Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
