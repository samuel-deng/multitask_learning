import numpy as np
import pickle
from tensorly.tenalg import mode_dot
from tensorly.tenalg import inner
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
        task_function[i] = np.random.randint(0, T)

    # TEST: Make sure <A(I, I, Z), X_i> + eps_i == R_{i, t(i)}
    # A(I_d1, I_d2, Z), just to check if generate_covariate_X is working okay
    A_test = np.zeros((d1, d2, T))
    for i in range(d1):
        for j in range(d2):
            for t in range(T):
                A_test[i][j][t] = A_Z_prod(A, Z, i, j, t)

    cov_X_list = []
    for i in range(len(X)):
        cov_X_list.append(generate_covariate_X(X[i], Y[task_function[i]], task_function[i], T)) 

    for i in range(len(X)):
        assert( np.abs(inner(cov_X_list[i], A_test) + noise[i][task_function[i]] - R[i][task_function[i]]) < 1e-6 ) # Check A(I_d, I_d2, Z) dot X gives back R_i
    return X, Y, Z, A, R, task_function, cov_X_list, A_test

# For testing
def generate_covariate_X(x, y, t, T):
    outer = np.outer(x, y)
    cov_X = np.zeros((len(x), len(y), T)) # d1 x d2 x T
    cov_X[:,:, t] = outer
    return cov_X

# For testing
def A_Z_prod(A, Z, i, j, t):
    cum_sum = 0
    for k in range(A.shape[2]):
        cum_sum += A[i][j][k] * Z[t][k]

    return cum_sum

if __name__ == "__main__":
    # Dataset parameters
    d1 = 100
    d2 = 100
    d3 = 100
    N = 2500
    T = 100
    r = 3

    # Generate synthetic data
    X, Y, Z, A, R, task_function, cov_X_list, A_test = generate_synthetic_data(d1, d2, d3, N, T, r)

    # Pickle the data to run tensor regression on
    pickle.dump(X, open("synthetic_data/X.pkl", "wb"))
    pickle.dump(Y, open("synthetic_data/Y.pkl", "wb"))
    pickle.dump(Z, open("synthetic_data/Z.pkl", "wb"))
    pickle.dump(A, open("synthetic_data/A.pkl", "wb"))
    pickle.dump(R, open("synthetic_data/R.pkl", "wb"))
    pickle.dump(task_function, open("synthetic_data/task_function.pkl", "wb"))
    pickle.dump(cov_X_list, open("synthetic_data/cov_X_list.pkl", "wb"))
    pickle.dump(A_test, open("synthetic_data/true_B.pkl", "wb"))