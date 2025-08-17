import math
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Function Parser
# ==============================
# Takes the string input (func_str) and evaluates it at point x.
# Allowed only "math" functions and variable x (safe eval).
def f(x):
    return eval(func_str, {"x": x, "math": math, "__builtins__": None})


# ==============================
# Numerical Derivative
# ==============================
# Uses central difference approximation to calculate derivative f'(x).
# h is a very small step size (default 1e-6).
def df(x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


# ==============================
# Newton-Raphson Method
# ==============================
def newton_raphson(x0, tol=1e-6, max_iter=200):
    print(f"\nStarting Newton-Raphson from x0 = {x0}")
    table = []       # To store iteration details for later display
    prev_x = None    # For error calculation

    # Iterative process
    for it in range(1, max_iter + 1):
        fx = f(x0)         # f(x)
        dfx = df(x0)       # f'(x)

        # Prevent division by zero if derivative vanishes
        if dfx == 0:
            print("Derivative is zero. Cannot proceed.")
            return None, []

        # Newton-Raphson update formula: x1 = x0 - f(x0)/f'(x0)
        x1 = x0 - fx / dfx
        fx1 = f(x1)        # f(x1) for next check

        # --------------------------
        # Error Calculation (Approx)
        # --------------------------
        if prev_x is not None:
            ea = abs((x1 - prev_x) / x1) * 100  # Relative % error
        else:
            ea = None   # First iteration has no error

        # --------------------------
        # Significant Digits (SD)
        # --------------------------
        # Rule: SD increases if error < 5*10^-sd
        if ea is not None and ea != 0:
            sd = 0
            threshold = 5 * 10**(-sd)
            while ea < threshold:   # Keep increasing SD until condition fails
                sd += 1
                threshold = 5 * 10**(-sd)
        else:
            sd = "∞" if ea == 0 else "-"  # Infinite SD if exact root found

        # Save iteration details
        table.append([it, x0, fx, dfx, x1, fx1, ea, sd])

        # Update for next iteration
        prev_x = x1

        # Stopping criteria: either f(x1) is close to 0 or successive x values converge
        if abs(fx1) < tol or abs(x1 - x0) < tol:
            break

        x0 = x1  # Move to next iteration

    # ==============================
    # Print Iteration Table
    # ==============================
    print(f"\n{'Iter':<6} {'x':>12} {'f(x)':>12} {'f\'(x)':>12} {'x_new':>12} {'f(x_new)':>12} {'Error(%)':>12} {'SD':>6}")
    for row in table:
        it, x, fx, dfx, x_new, fx_new, ea, sd = row
        ea_str = f"{ea:>12.6f}" if ea is not None else " " * 12
        print(f"{it:<6} {x:>12.6f} {fx:>12.6f} {dfx:>12.6f} {x_new:>12.6f} {fx_new:>12.6f} {ea_str} {str(sd):>6}")

    # Print final root result
    print(f"\nApproximate Root found at x = {table[-1][4]:.6f} after {table[-1][0]} iterations")
    return table[-1][4], table


# ==============================
# Input Section
# ==============================
func_str = input("Enter function in terms of x (e.g. x^3 - x - 2): ").replace("^", "**")

try:
    # Get user inputs with defaults for tolerance and iteration count
    x0 = float(input("Enter initial guess x0: "))
    tol_input = input("Enter tolerance (default 1e-6): ")
    tol = float(tol_input) if tol_input else 1e-6
    iter_input = input("Enter maximum iterations (default 200): ")
    max_iter = int(iter_input) if iter_input else 200
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()


# ==============================
# Run Newton-Raphson
# ==============================
root, table = newton_raphson(x0, tol, max_iter)

if root is not None:
    print(f"\nFinal Approximate root: {root:.6f}")

    # ==============================
    # Plotting Section
    # ==============================
    x_vals = np.linspace(root - 5, root + 5, 400)  # Range around root
    y_vals = [f(x) for x in x_vals]

    # Points used during iterations
    x_points = [row[1] for row in table]
    y_points = [f(x) for x in x_points]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {func_str}", color='blue')
    plt.axhline(0, color='black', linewidth=0.5)  # X-axis

    # Iteration points and final root marker
    plt.scatter(x_points, y_points, color='red', label='Approximations', zorder=5)
    plt.scatter([root], [f(root)], color='green', s=100, label='Final Root', edgecolors='black')

    # Draw tangent lines at each approximation point
    for row in table:
        x = row[1]
        slope = df(x)
        y = f(x)
        tangent_x = np.linspace(x - 1, x + 1, 10)  # small range around x
        tangent_y = slope * (tangent_x - x) + y
        plt.plot(tangent_x, tangent_y, color='gray', linestyle='--', alpha=0.5)

    # Plot styling
    plt.title("Newton-Raphson Method Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
