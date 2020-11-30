import numpy as np
from generate_data import generate_synthetic_data
from tensor_regression import generate_covariate_X, grad_descent

if __name__ == "__main__":
    # Dataset parameters
    d1 = 100
    d2 = 100
    d3 = 100
    N = 100
    T = 100
    r = 3

    # Generate dataset
    X, Y, Z, A, R, task_function = generate_synthetic_data(d1, d2, d3, N, T, r)

    # Tensor regression
    eta = 0.1
    eps = 0.1
    lambd = 0.1
    B = grad_descent(R, X, Y, T, eta, eps, lambd, task_function) # Works! Gets stuck at around ~0.3 loss

    # Tensor decomposition

