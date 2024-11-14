import numpy as np
from functions import gradient, armijo_line_search


def steepest_descent(f, x0, epsilon=1e-10, max_iter=10000):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = gradient(f, x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < epsilon:
            break  
        p = -grad / grad_norm  # Normalized descent direction
        alpha = armijo_line_search(f, x, p, grad)
        if alpha <  epsilon:
            break
        x = x + alpha * p
        print("Iteration: ",i)
        print("Grad: ",grad_norm)
        print("Alpha: ",alpha)
        print("X = ",x)
        print("f(x): ",f(x))
    return x

def f1(x):
    return x[0]**2 + x[1]**2 + x[2]**2

x0_f1 = np.array([1.0, 1.0, 1.0])

def f2(x):
    return x[0]**2 + 2 * x[1]**2 - 2 * x[0] * x[1] - 2 * x[1]

x0_f2 = np.array([0.0, 0.0])

def f3(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

x0_f3 = np.array([-1.2, 1.0])

def f4(x):
    return (x[0] + x[1])**4 + x[1]**2

x0_f4 = np.array([2.0, -2.0])

def f51(x, c=1):
    return (x[0] - 1)**2 + (x[1] - 1)**2 + c * (x[0]**2 + x[1]**2 - 0.25)**2

x0_f5 = np.array([1.0, -1.0])


sol_steep = steepest_descent(f1,x0_f1)
print("f(x0):", f1(x0_f1))
print("Solución:", sol_steep)