import numpy as np
from generate_data import generate_synthetic_data
from tensor_regression import generate_covariate_X, grad_descent

if __name__ == "__main__":
    # Dataset parameters
    d1 = 10
    d2 = 10
    d3 = 10
    N = 50
    T = 50
    r = 3

    # Generate dataset
    X, Y, Z, A, R = generate_synthetic_data(d1, d2, d3, N, T, r)

    # Tensor regression
    eta = 0.1
    eps = 0.1
    lambd = 0.1
    B = grad_descent(R, X, Y, T, eta, eps, lambd)

    # Tensor decomposition

