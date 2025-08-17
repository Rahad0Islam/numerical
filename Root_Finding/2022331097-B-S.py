import math
import matplotlib.pyplot as plt
import numpy as np

# Function parser (safe eval: only allows math functions & variable x)
def f(x):
    return eval(func_str, {"x": x, "math": math, "__builtins__": None})

# ---------------- Bisection Method Implementation ----------------
def bisection(lower_bound, upper_bound, tolerance=1e-6, max_iterations=200):
    """
    Finds a root of f(x) in the interval [lower_bound, upper_bound] using the bisection method.
    """
    # Check that f(a) and f(b) have opposite signs (Intermediate Value Theorem condition)
    if f(lower_bound) * f(upper_bound) >= 0:
        print("Bisection method fails: f(a) and f(b) must have opposite signs.")
        return None, []

    print(f"\nValid interval: [{lower_bound}, {upper_bound}]")
    table = []      # To store iteration data for printing later
    previous_mid = None  # Store previous midpoint for error calculation

    # Main iteration loop
    for iteration in range(1, max_iterations + 1):
        # Midpoint
        midpoint = (lower_bound + upper_bound) / 2
        f_a = f(lower_bound)
        f_b = f(upper_bound)
        f_mid = f(midpoint)

        # Approximate error (percentage)
        if previous_mid is not None:
            error = abs((midpoint - previous_mid) / midpoint) * 100
        else:
            error = None

        # Significant digits check
        if error is not None and error != 0:
            sig_digits = 0
            threshold = 5 * 10**(-sig_digits)
            while error < threshold:
                sig_digits += 1
                threshold = 5 * 10**(-sig_digits)
        else:
            sig_digits = "∞" if error == 0 else "-"

        # Store iteration data
        table.append([iteration, lower_bound, upper_bound, f_a, f_b, midpoint, f_mid, error, sig_digits])

        # Update for next iteration
        previous_mid = midpoint

        # Check stopping conditions: function close to zero or interval small enough
        if abs(f_mid) < tolerance or abs(upper_bound - lower_bound) < tolerance:
            break

        # Update interval based on sign of f(mid)
        if f_a * f_mid > 0:
            lower_bound = midpoint  # Root lies in [mid, b]
        else:
            upper_bound = midpoint  # Root lies in [a, mid]

    # Print iteration table
    print(f"\n{'Iter':<6} {'a':>12} {'b':>12} {'f(a)':>12} {'f(b)':>12} "
          f"{'mid':>12} {'f(mid)':>12} {'Error(%)':>12} {'SD':>6}")

    for row in table:
        it, a, b, fa, fb, c, fc, err, sd = row
        err_str = f"{err:>12.6f}" if err is not None else " " * 12
        print(f"{it:<6} {a:>12.6f} {b:>12.6f} {fa:>12.6f} {fb:>12.6f} "
              f"{c:>12.6f} {fc:>12.6f} {err_str} {str(sd):>6}")

    # Print final result
    print(f"\nApproximate Root found at x = {table[-1][5]:.6f} "
          f"after {table[-1][0]} iterations")
    return table[-1][5], table


# ---------------- Input Section ----------------
func_str = input("Enter function in terms of x (e.g. x^3 - x + 5): ").replace("^", "**")

try:
    lower = float(input("Enter lower bound a: "))
    upper = float(input("Enter upper bound b: "))
    tol_input = input("Enter tolerance (default 1e-6): ")
    tolerance = float(tol_input) if tol_input else 1e-6
    iter_input = input("Enter maximum iterations (default 200): ")
    max_iter = int(iter_input) if iter_input else 200
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

# ---------------- Run Bisection ----------------
root, iterations_table = bisection(lower, upper, tolerance, max_iter)

if root is not None:
    print(f"\nFinal Approximate Root: {root:.6f}")

    # ---------------- Plotting ----------------
    x_vals = np.linspace(lower, upper, 400)
    y_vals = [f(x) for x in x_vals]

    # Midpoints & their function values
    c_vals = [row[5] for row in iterations_table]
    fc_vals = [row[6] for row in iterations_table]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {func_str}", color='blue')
    plt.axhline(0, color='black', linewidth=0.5)

    # Plot midpoints and final root
    plt.scatter(c_vals, fc_vals, color='red', label='Midpoints (c)', zorder=5)
    plt.scatter([root], [f(root)], color='green', s=100,
                label='Approximate Root', edgecolors='black')

    plt.title("Bisection Method Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
