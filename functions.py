import numpy as np

# Gradient approximation by finite differences
def gradient(f, x, eps=1e-5):
    
    grad = np.zeros_like(x, dtype=float)
    
    for i in range(len(x)):
        x_eps = np.array(x, dtype=float)
        x_eps[i] += eps
        grad[i] = (f(x_eps) - f(x)) / eps
    return grad

def hessian(f, x, eps=1e-5):
    
    n = len(x)
    hess = np.zeros((n, n), dtype=float)  # Initialize an n x n Hessian matrix

    for i in range(n):
        x_eps = np.array(x, dtype=float)
        x_eps[i] += eps
        grad_plus = gradient(f, x_eps)  # Gradient at x + eps in the i-th direction
        
        x_eps[i] -= 2 * eps
        grad_minus = gradient(f, x_eps)  # Gradient at x - eps in the i-th direction

        hess[i, :] = (grad_plus - grad_minus) / (2 * eps)

    return hess

# Calculate the value of alpha by using the armijo condition

def armijo_line_search(f, x, p, grad, alpha_init=1.0, c=1e-4, max_iter=100):
    
    alpha = alpha_init
    
    for _ in range(max_iter):
        if f(x + alpha * p) <= f(x) + c * alpha * np.dot(grad, p):
            return alpha
        alpha /= 2  # Reduce alpha by half each time
    print("Warning: Armijo line search did not converge.")
    return alpha

# Backtracking line search 

def line_search(f, x, p, alpha=1.0, beta=0.9, tol=1e-5, max_iter=1000):
    """
    Backtracking line search to find an appropriate step size alpha.
    """
    x = np.array(x, dtype=float)
    grad = gradient(f, x)
    grad_norm_sq = np.dot(grad, grad)
    i = 0
    while f(x + alpha * p) <= f(x) + tol * alpha * np.dot(gradient(f, x), p):
        alpha *= beta  
        i += 1  
        if i >= max_iter:
            print("Line search failed to converge.")
            break
    return alpha