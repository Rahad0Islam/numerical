import matplotlib.pyplot as plt
import sympy as sp
import math

# -----------------------------------------------------
# Euler Method
# -----------------------------------------------------
def euler_method(f, y0, x0, xf, h):
    """
    Euler's Method for solving ODE y' = f(x, y)
    Returns lists of x and y values
    """
    x, y = x0, y0
    x_vals, y_vals = [x], [y]

    while x < xf:
        step = min(h, xf - x)  # avoid overshooting final x
        y += f(x, y) * step    # Euler formula
        x += step
        x_vals.append(x)
        y_vals.append(y)

    return x_vals, y_vals

# -----------------------------------------------------
# Heun Method (Improved Euler)
# -----------------------------------------------------
def heun_method(f, y0, x0, xf, h):
    x, y = x0, y0
    x_vals, y_vals = [x], [y]

    while x < xf:
        step = min(h, xf - x)
        k1 = f(x, y)
        k2 = f(x + step, y + k1 * step)
        y += 0.5 * (k1 + k2) * step
        x += step
        x_vals.append(x)
        y_vals.append(y)

    return x_vals, y_vals

# -----------------------------------------------------
# Midpoint Method (2nd-order RK)
# -----------------------------------------------------
def midpoint_method(f, y0, x0, xf, h):
    x, y = x0, y0
    x_vals, y_vals = [x], [y]

    while x < xf:
        step = min(h, xf - x)
        k1 = f(x, y)
        k2 = f(x + 0.5 * step, y + 0.5 * k1 * step)
        y += k2 * step
        x += step
        x_vals.append(x)
        y_vals.append(y)

    return x_vals, y_vals

# -----------------------------------------------------
# Ralston Method (2nd-order RK with weighted slopes)
# -----------------------------------------------------
def ralston_method(f, y0, x0, xf, h):
    x, y = x0, y0
    x_vals, y_vals = [x], [y]

    while x < xf:
        step = min(h, xf - x)
        k1 = f(x, y)
        k2 = f(x + 0.75 * step, y + 0.75 * k1 * step)
        y += (1/3 * k1 + 2/3 * k2) * step
        x += step
        x_vals.append(x)
        y_vals.append(y)

    return x_vals, y_vals

# -----------------------------------------------------
# Compare step sizes
# -----------------------------------------------------
def compare_step_sizes(f, y0, x0, xf, step_sizes):
    print("\nPerformance Comparison Across Step Sizes:")
    print(f"{'h':<8}{'Euler':<12}{'Heun':<12}{'Midpoint':<12}{'Ralston':<12}")
    for h in step_sizes:
        e = euler_method(f, y0, x0, xf, h)[1][-1]       # last y value
        hn = heun_method(f, y0, x0, xf, h)[1][-1]
        mp = midpoint_method(f, y0, x0, xf, h)[1][-1]
        rl = ralston_method(f, y0, x0, xf, h)[1][-1]
        print(f"{h:<8.5f}{e:<12.5f}{hn:<12.5f}{mp:<12.5f}{rl:<12.5f}")

# -----------------------------------------------------
# Main function
# -----------------------------------------------------
def main():
    try:
        # Step 1: Input ODE safely
        func_str = input("Enter the ODE function f(x, y) (polynomial in x and y): ")

        # Step 2: Input initial conditions and parameters
        x0 = float(input("Enter initial x0: "))
        y0 = float(input("Enter initial y0: "))
        xf = float(input("Enter final xf: "))
        h = float(input("Enter step size h: "))

        # Step 3: Convert string to safe function using SymPy
        x_sym, y_sym = sp.symbols("x y")
        try:
            sym_expr = sp.sympify(func_str.replace("^", "**"))
        except sp.SympifyError:
            print("Invalid expression. Please enter a polynomial in x and y.")
            return

        f = sp.lambdify((x_sym, y_sym), sym_expr, modules=["math"])

        # Step 4: Solve ODE using all four methods
        x_e, y_e = euler_method(f, y0, x0, xf, h)
        x_h, y_h = heun_method(f, y0, x0, xf, h)
        x_m, y_m = midpoint_method(f, y0, x0, xf, h)
        x_r, y_r = ralston_method(f, y0, x0, xf, h)

        # Step 5: Print results at final x
        print(f"\nResults at x = {xf}:")
        print(f"Euler's Method:   {y_e[-1]:.5f}")
        print(f"Heun's Method:    {y_h[-1]:.5f}")
        print(f"Midpoint Method:  {y_m[-1]:.5f}")
        print(f"Ralston's Method: {y_r[-1]:.5f}")

        # Step 6: Compare performance across multiple step sizes
        step_sizes = [h / (2**i) for i in range(5)]
        compare_step_sizes(f, y0, x0, xf, step_sizes)

        # Step 7: Plot all methods
        plt.figure(figsize=(10, 6))
        plt.plot(x_e, y_e, "ro-", label="Euler")        # red circles
        plt.plot(x_h, y_h, "bs-", label="Heun")         # blue squares
        plt.plot(x_m, y_m, "g^-", label="Midpoint")     # green triangles
        plt.plot(x_r, y_r, "m*-", label="Ralston")      # magenta stars
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("ODE Solution Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()

    except ValueError:
        print("Invalid numerical input.")
    except Exception as e:
        print(f"Error: {e}")

# -----------------------------------------------------
# Run main
# -----------------------------------------------------
if __name__ == "__main__":
    main()
