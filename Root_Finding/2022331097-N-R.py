import math

poly_nom = input("Enter the polynomial Equation (e.g., x^3 - 4*x - 20): ")
poly_nom = poly_nom.replace("^", "**")

def func(x):
    return eval(poly_nom, {"x": x, "math": math, "__builtins__": None})

def deFunc(x, h=1e-6):
    return (func(x + h) - func(x - h)) / (2 * h)

def Newton_raph(x0, tol=1e-6, max_it=1000):
    print(f"\nStarting Newton-Raphson from x0 = {x0}\n")
    it = 1
    print("\nit\t    x0\t\t   f(x0)\t f'(x0)\t\t   x1\t\t  f(x1)")
    
    while it <= max_it:
        fx = func(x0)
        dfx = deFunc(x0)
        
        if dfx == 0:
            print("\nDerivative is zero. Cannot proceed.\n")
            return 
        
        x1 = x0 - fx / dfx
        fx1 = func(x1)
        
        print(f"{it}\t {x0:.6f}\t {fx:.6f}\t {dfx:.6f}\t {x1:.6f}\t {fx1:.6f}")
        
        if abs(x1 - x0) < tol or abs(fx1) < tol:
            print(f"\nRoot is approximately: {x1:.6f}")
            break
        
        x0 = x1
        it += 1
    else:
        print("\nMaximum iterations reached without convergence.")

x0 = float(input("Enter initial guess x0: "))

tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
max_it = int(input("Enter maximum iterations (default 100): ") or 100)

Newton_raph(x0, tol, max_it)
