import numpy as np
import pickle
import argparse
import time
from tensorly import norm
from tensorly.tenalg import mode_dot
from tensorly.tenalg import inner
from tensorly.tenalg import multi_mode_dot
from tensorly.cp_tensor import cp_to_tensor
from tensorly.random import random_cp
from functools import reduce

def outer(*vs):
    return reduce(np.multiply.outer, vs)

def generate_responses(A, X, Y, Z, T):
    R = [multi_mode_dot(mode_dot(A, X, mode=0), [Y[t], Z[t]], modes=[1, 2]) for t in range(T)]
    R = np.asarray(R).T
    return R

# Only generates X, Y, Z data
def generate_training_data(d1, d2, d3, N, T ):

    # Generate user feature vectors (X)
    user_mu = 0
    user_sigma = 1/np.sqrt(d1)
    X = user_sigma * np.random.randn(N, d1) + user_mu # From N(0, 1)

    # Generate task feature vectors (Y, Z)
    Y_mu = 0
    Y_sigma = 1/np.sqrt(d2)
    Z_mu = 0
    Z_sigma = 1/np.sqrt(d3)
    Y = Y_sigma * np.random.randn(T, d2) + Y_mu
    Z = Z_sigma * np.random.randn(T, d3) + Z_mu
    return X, Y, Z

# Only generates underlying system tensor (A), with rank r
def generate_A_tensor(d1, d2, d3, r ):

    # Generate underlying tensor (A), rank r
    (_, factors) = random_cp((d1, d2, d3), full=False, rank=r, orthogonal=True, normalise_factors=True)
    #weights = np.random.uniform(low=1.0, high=10.0, size=(r))
    weights = np.ones(r)
    A = cp_to_tensor((weights, factors))
    return A

def generate_synthetic_data(d1, d2, d3, N, T, r, sigma ):
    start = time.time()

    # Generate training data (X, Y, Z)
    X, Y, Z = generate_training_data(d1, d2, d3, N, T )

    # Generate Gaussian noise
    noise_mu, noise_sigma = 0, sigma
    noise = np.random.normal(noise_mu, noise_sigma, (N, T))

    # Generate underlying tensor (A), rank r
    A = generate_A_tensor(d1, d2, d3, r )

    # Generate responses
    R = generate_responses(A, X, Y, Z, T)
    R += noise

    # Task function t(i)
    # Assign each user with a task uniformly at random
    task_function = np.random.randint(0, T, size=N)

    # Find the true B ( <A(I, I, Z), X_i> ) that we'll estimate with tensor regression
    true_B = mode_dot(A, Z, mode=2)

    # Generate covariate_X (not the full sparse tensor, just the slices)
    # cov_X[i] is equiv. to the t(i) slice of the i'th covariate X_i tensor
    Y_ti = Y[task_function]
    cov_X = np.einsum('bi,bo->bio', X, Y_ti)
    end = time.time()
    print("Time to generate data: {}".format(end - start))
    return X, Y, Z, A, R, task_function, cov_X, true_B
