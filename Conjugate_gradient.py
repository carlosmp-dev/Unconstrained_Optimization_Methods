import numpy as np
from functions import gradient, armijo_line_search

def conjugated_gradient(f, x0, epsilon=1e-8,max_iter = 1000):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad_xk = gradient(f, x)
        dk = - grad_xk
        grad_norm_xk = np.linalg.norm(grad_xk)
        alpha = armijo_line_search(f , x , dk, grad_xk)
        x =  x + alpha * dk
        grad_xk1 = gradient(f,x)
        grad_norm_xk1 = np.linalg.norm(grad_xk1)
        beta = ((grad_xk1)**2)/((grad_norm_xk)**2)
        dk = grad_xk1 + beta * dk
        if grad_norm_xk1 / (1 + abs(f(x))) <  epsilon:
            print("Iteration: ",i)
            print("Grad: ",grad_xk1)
            print("Alpha: ",alpha)
            print("X = ",x)
            print("f(x): ",f(x))
            return x
        
        print("Iteration: ",i)
        print("Grad: ",grad_xk1)
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


sol_conj = conjugated_gradient(f1,x0_f1)
print("f(x0):", f1(x0_f1))
print("Solución:", sol_conj)