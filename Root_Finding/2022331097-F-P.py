import math

poly_string = input("Enter polynomial Equation (e.g., x^3 - 4*x - 20): ")
poly_string = poly_string.replace("^", "**")

def func(x):
    return eval(poly_string, {"x": x, "math": math, "__builtins__": None})

def false_pos(a, b, tol=1e-6, max_it=1000):
    fa = func(a)
    fb = func(b)
    
    if fa * fb > 0:
        print("\nError: f(a) and f(b) must have opposite signs. Method cannot proceed.")
        return

    print(f"\nUsing interval: [{a}, {b}]\n")
    print("it\t  a\t\t  b\t\t f(a)\t\t f(b)\t\t c\t\t f(c)")
    it = 1
    
    while it <= max_it:
        fa = func(a)
        fb = func(b)
        if fb - fa == 0:
            print("Division by zero encountered.")
            return
        
        c = (a * fb - b * fa) / (fb - fa)
        fc = func(c)
        
        print(f"{it}\t {a:.6f}\t {b:.6f}\t {fa:.6f}\t {fb:.6f}\t {c:.6f}\t {fc:.6f}")
        
        if abs(fc) < tol or abs(b - a) < tol:
            break
        
        if fa * fc > 0:
            a = c
        else:
            b = c
        
        it += 1
    
    print("\nRoot is:", c)

a = float(input("Enter lower bound a: "))
b = float(input("Enter upper bound b: "))

tol = float(input("Enter tolerance (e.g. 1e-6): "))
max_it = int(input("Enter maximum iterations (e.g. 100): "))

false_pos(a, b, tol, max_it)
