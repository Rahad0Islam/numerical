import math
import matplotlib.pyplot as plt
import numpy as np

# Function parser: evaluates a function string entered by the user
def f(x):
    # We use eval() safely by only allowing 'x' and 'math' functions
    return eval(func_str, {"x": x, "math": math, "__builtins__": None})

# False Position (Regula Falsi) Method implementation
def false_position(a, b, tol=1e-6, max_iter=100):
    # Step 1: Check if the initial interval is valid
    # The function must have opposite signs at a and b
    if f(a) * f(b) >= 0:
        print("False Position method fails: f(a) and f(b) must have opposite signs.")
        return None, []

    print(f"\nValid interval: [{a}, {b}]")
    table = []       # To store iteration data
    prev_c = None    # Store previous approximation (for error calculation)

    # Step 2: Iteration loop
    for it in range(1, max_iter + 1):
        fa = f(a)
        fb = f(b)

        # Avoid division by zero (when f(a) ≈ f(b))
        if fb - fa == 0:
            print("Division by zero encountered.")
            break

        # False Position formula:
        # c = (a*f(b) - b*f(a)) / (f(b) - f(a))
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)

        # Step 3: Compute approximate relative error
        if prev_c is not None:
            ea = abs((c - prev_c) / c) * 100  # % error
        else:
            ea = None   # No error for first iteration

        # Step 4: Determine significant digits (SD) based on error
        if ea is not None and ea != 0:
            sd = 0
            threshold = 5 * 10**(-sd)
            while ea < threshold:  # Keep counting SD until error exceeds threshold
                sd += 1
                threshold = 5 * 10**(-sd)
        else:
            sd = "∞" if ea == 0 else "-"   # Infinite SD if exact root found

        # Store results in the iteration table
        table.append([it, a, b, fa, fb, c, fc, ea, sd])
        prev_c = c  # Update previous approximation

        # Step 5: Check stopping conditions
        if abs(fc) < tol or abs(b - a) < tol:
            break

        # Step 6: Update interval [a, b]
        # Keep the subinterval where the root lies
        if fa * fc < 0:
            b = c
        else:
            a = c

    # Step 7: Print results in tabular format
    print(f"\n{'Iter':<6} {'a':>12} {'b':>12} {'f(a)':>12} {'f(b)':>12} {'c':>12} {'f(c)':>12} {'Error(%)':>12} {'SD':>6}")

    for row in table:
        it, a, b, fa, fb, c, fc, ea, sd = row
        ea_str = f"{ea:>12.6f}" if ea is not None else " " * 12
        print(f"{it:<6} {a:>12.6f} {b:>12.6f} {fa:>12.6f} {fb:>12.6f} {c:>12.6f} {fc:>12.6f} {ea_str} {str(sd):>6}")

    # Final root after all iterations
    print(f"\nRoot found at x = {table[-1][5]:.6f} after {table[-1][0]} iterations")
    return table[-1][5], table


# ---------------- MAIN PROGRAM ----------------

# Step A: Get function string from user
func_str = input("Enter function in terms of x (e.g. x^2 - x - 20): ").replace("^", "**")

try:
    # Step B: Get interval and parameters
    a = float(input("Enter lower bound: "))
    b = float(input("Enter upper bound: "))
    tol_input = input("Enter tolerance (default 1e-6): ")
    tol = float(tol_input) if tol_input else 1e-6
    iter_input = input("Enter maximum iterations (default 200): ")
    max_iter = int(iter_input) if iter_input else 200
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

# Step C: Run False Position Method
root, table = false_position(a, b, tol, max_iter)

if root is not None:
    print(f"\nFinal Approximate root: {root:.6f}")

    # Step D: Visualization using matplotlib
    x_vals = np.linspace(a, b, 400)      # Generate x values in range
    y_vals = [f(x) for x in x_vals]      # Compute f(x) for plotting

    # Extract approximations from the iteration table
    c_vals = [row[5] for row in table]
    fc_vals = [row[6] for row in table]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {func_str}", color='blue')
    plt.axhline(0, color='black', linewidth=0.5)

    # Mark approximation points
    plt.scatter(c_vals, fc_vals, color='red', label='Approximations (c)', zorder=5)
    plt.scatter([root], [f(root)], color='green', s=100, label='Final Root', edgecolors='black')

    plt.title("False Position Method Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
