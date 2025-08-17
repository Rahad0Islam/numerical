import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Symbolic Setup
# ==============================
x = sp.symbols('x')
func_str = input("Enter function in terms of x (e.g. x**2 - 4*x + 3): ")
f_sym = sp.sympify(func_str)        # Convert string to SymPy expression
f_prime = sp.diff(f_sym, x)         # Symbolic derivative

print(f"\nOriginal function: f(x) = {f_sym}")
print(f"Derivative function: f'(x) = {f_prime}")

# Convert to numeric functions for iteration
def f_num(x_val):
    return float(f_sym.evalf(subs={x: x_val}))

def df_num(x_val):
    return float(f_prime.evalf(subs={x: x_val}))

# ==============================
# Newton-Raphson Method
# ==============================
def newton_raphson(x0, tol=1e-6, max_iter=200):
    print(f"\nStarting Newton-Raphson from x0 = {x0}")
    table = []
    prev_x = None

    for it in range(1, max_iter + 1):
        fx = f_num(x0)
        dfx = df_num(x0)

        if dfx == 0:
            print("Derivative is zero. Cannot proceed.")
            return None, []

        x1 = x0 - fx / dfx
        fx1 = f_num(x1)

        # Approximate error
        if prev_x is not None:
            ea = abs((x1 - prev_x) / x1) * 100
        else:
            ea = None

        # Significant digits
        if ea is not None and ea != 0:
            sd = 0
            threshold = 5 * 10**(-sd)
            while ea < threshold:
                sd += 1
                threshold = 5 * 10**(-sd)
        else:
            sd = "∞" if ea == 0 else "-"

        table.append([it, x0, fx, dfx, x1, fx1, ea, sd])
        prev_x = x1

        if abs(fx1) < tol or abs(x1 - x0) < tol:
            break

        x0 = x1

    # Print iteration table
    print(f"\n{'Iter':<6} {'x':>12} {'f(x)':>12} {'f\'(x)':>12} {'x_new':>12} {'f(x_new)':>12} {'Error(%)':>12} {'SD':>6}")
    for row in table:
        it, x_val, fx_val, dfx_val, x_new, fx_new, ea, sd = row
        ea_str = f"{ea:>12.6f}" if ea is not None else " " * 12
        print(f"{it:<6} {x_val:>12.6f} {fx_val:>12.6f} {dfx_val:>12.6f} {x_new:>12.6f} {fx_new:>12.6f} {ea_str} {str(sd):>6}")

    print(f"\nApproximate Root found at x = {table[-1][4]:.6f} after {table[-1][0]} iterations")
    return table[-1][4], table

# ==============================
# Input initial guess and parameters
# ==============================
try:
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
    # Plotting
    # ==============================
    x_vals = np.linspace(root - 5, root + 5, 400)
    y_vals = [f_num(val) for val in x_vals]

    x_points = [row[1] for row in table]
    y_points = [f_num(val) for val in x_points]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {f_sym}", color='blue')
    plt.axhline(0, color='black', linewidth=0.5)

    # Approximations and final root
    plt.scatter(x_points, y_points, color='red', label='Approximations', zorder=5)
    plt.scatter([root], [f_num(root)], color='green', s=100, label='Final Root', edgecolors='black')

    # Tangent lines at each iteration point
    for row in table:
        x_val = row[1]
        slope = df_num(x_val)
        y_val = f_num(x_val)
        tangent_x = np.linspace(x_val - 1, x_val + 1, 10)
        tangent_y = slope * (tangent_x - x_val) + y_val
        plt.plot(tangent_x, tangent_y, color='gray', linestyle='--', alpha=0.5)

    plt.title("Newton-Raphson Method Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
