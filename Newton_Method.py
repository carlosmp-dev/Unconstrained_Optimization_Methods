import numpy as np
from functions import gradient, hessian

def newton_method(f, x0, epsilon=1e-5, max_iter=100):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = gradient(f, x)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < epsilon:
            break
        
        hess = hessian(f, x)
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            hess_inv = np.linalg.pinv(hess)

        p = -hess_inv @ grad
        x = x + p
        
        print("Iteration: ",i)
        print("Grad: ",grad_norm)
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


sol_newton = newton_method(f1,x0_f1)
print("f(x0):", f1(x0_f1))
print("Solución:", sol_newton)