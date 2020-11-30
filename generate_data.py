import numpy as np
from tensorly.tenalg import multi_mode_dot
from tensorly.random import random_cp
from functools import reduce

def outer(*vs):
    return reduce(np.multiply.outer, vs)

def generate_synthetic_data(d1, d2, d3, N, T, r):
    # Generate user feature vectors (X)
    user_mu = 0
    user_sigma = 1 
    X = user_sigma * np.random.randn(N, d1) + user_mu # From N(0, 1)

    # Generate task feature vectors (Y, Z)
    Y_mu = 0
    Y_sigma = 1
    Z_mu = 0
    Z_sigma = 1
    Y = Y_sigma * np.random.randn(T, d2) + Y_mu
    Z = Z_sigma * np.random.randn(T, d3) + Z_mu

    # Generate Gaussian noise
    noise_mu, noise_sigma = 0, 0.1
    noise = np.random.normal(noise_mu, noise_sigma, (N, T))

    # Generate underlying tensor (A), rank r
    A = random_cp((d1, d2, d3), full=True, rank=r, orthogonal=True)

    # Generate responses
    R = np.zeros((N, T))
    for n in range(N):
        for t in range(T):
            A_prod = multi_mode_dot(A, [X[n], Y[t], Z[t]])
            R[n][t] = A_prod + noise[n][t] 

    # Task function t(i)
    # Assign each user with a task uniformly at random
    task_function = dict()
    for i in range(N):
        task_function[i] = i

    return X, Y, Z, A, R, task_function