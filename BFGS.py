import numpy as np
from functions import gradient, armijo_line_search

def BFGS(f, x0, epsilon=1e-8, max_iter=1000):
    xk = np.array(x0, dtype=float)
    B = np.eye(len(xk))
    for i in range(max_iter):
        grad_xk = gradient(f, xk)
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            B_inv = np.linalg.pinv(B)
        p = - B_inv @ grad_xk
        alpha = armijo_line_search(f , xk , p , grad_xk)
        xk1 = xk + alpha * p
        grad_xk1 =  gradient(f, xk1)
        s = xk1 - xk
        y =  grad_xk1 - grad_xk
        if np.dot(s.T , y) > epsilon:
            B = B + np.outer(y , y) / np.dot(y , s) - np.dot(B @ np.outer(s , s) , B) / np.dot(s , B @ s)
        xk = xk1
        if np.linalg.norm(grad_xk) / (1 + abs(f(xk))) <  epsilon:
            return xk

        print("Iteration: ",i)
        print("Grad: ",grad_xk1)
        print("Alpha: ",alpha)
        print("X = ",xk)
        print("f(x): ",f(xk))
    return xk



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


sol_BFGS = BFGS(f1,x0_f1)
print("f(x0):", f1(x0_f1))
print("Solución:", sol_BFGS)