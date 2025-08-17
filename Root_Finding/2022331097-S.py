import math


poly_nom = input("Enter the polynomial Equation (e.g., x^3 - 4*x - 20): ")
poly_nom = poly_nom.replace("^", "**")

def func(x):
    return eval(poly_nom, {"x": x, "math": math, "__builtins__": None})

def secant(x0, x1, tol=1e-6, max_it=1000):
    print(f"\nStarting Secant Method from x0 = {x0}, x1 = {x1}\n")
    print("it\t   x0\t\t   x1\t\t  f(x0)\t\t  f(x1)\t\t   x2\t\t  f(x2)")
    
    it = 1
    while it <= max_it:
        fx0 = func(x0)
        fx1 = func(x1)
        
        if fx1 - fx0 == 0:
            print("\nDivision by zero encountered. Cannot proceed.")
            return
        
        x2 = x1 - ((x1 - x0) / (fx1 - fx0)) * fx1
        fx2 = func(x2)
        
        print(f"{it}\t {x0:.6f}\t {x1:.6f}\t {fx0:.6f}\t {fx1:.6f}\t {x2:.6f}\t {fx2:.6f}")
        
        if abs(x2 - x1) < tol or abs(fx2) < tol:
            print(f"\nRoot is approximately: {x2:.6f}")
            return
        
        x0, x1 = x1, x2
        it += 1
    
    print("\nMaximum iterations reached without convergence.")
    print(f"Last approximation: {x2:.6f}")
    
x0 = float(input("Enter first initial guess x0: "))
x1 = float(input("Enter second initial guess x1: "))


tol = float(input("Enter tolerance (e.g. 1e-6): "))
max_it = int(input("Enter maximum iterations (e.g.  100): "))

secant(x0, x1, tol, max_it)
