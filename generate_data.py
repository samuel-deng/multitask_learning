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
    weights = np.random.uniform(low=1.0, high=10.0, size=(r))
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
    # R = [multi_mode_dot(mode_dot(A, X, mode=0), [Y[t], Z[t]], modes=[1, 2]) for t in range(T)]
    # R = np.asarray(R).T + noise

    # Task function t(i)
    # Assign each user with a task uniformly at random
    task_function = np.random.randint(0, T, size=N)

    # Find the true B ( <A(I, I, Z), X_i> ) that we'll estimate with tensor regression
    true_B = mode_dot(A, Z, mode=2)

    # Generate cov_X_list of actual covariate X tensors
    #cov_X_list = []
    #for i in range(N):
    #    cov_X_list.append(generate_covariate_X(X[i], Y[task_function[i]], task_function[i], T))

    # Generate covariate_X (not the full sparse tensor, just the slices)
    # cov_X[i] is equiv. to the t(i) slice of the i'th covariate X_i tensor
    Y_ti = Y[task_function]
    cov_X = np.einsum('bi,bo->bio', X, Y_ti)
    # return X, Y, Z, A, R, task_function, cov_X, cov_X_list, true_B
    end = time.time()
    print("Time to generate data: {}".format(end - start))
    return X, Y, Z, A, R, task_function, cov_X, true_B

# For testing
#def generate_covariate_X(x, y, t, T):
#    outer = np.outer(x, y)
#    cov_X = np.zeros((len(x), len(y), T)) # d1 x d2 x T
#    cov_X[:,:, t] = outer
#    return cov_X

# For testing
#def A_Z_prod(A, Z, i, j, t):
#    cum_sum = 0
#    for k in range(A.shape[2]):
#        cum_sum += A[i][j][k] * Z[t][k]

#    return cum_sum

if __name__ == "__main__":
    # Parse arguments from command line
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", help="First dimension (d1).")
    parser.add_argument("--d2", help="Second dimension (d2).")
    parser.add_argument("--d3", help="Third dimension (d3).")
    parser.add_argument("--N", help= "Number of users/examples (N).")
    parser.add_argument("--T", help="Number of tasks (T).")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
    parser.add_argument("--sigma", help="Std. dev. of the noise.")

    # Parse args (otherwise set defaults)
    args = parser.parse_args()
    if args.d1:
        d1 = int(args.d1)
    else:
        d1 = 100
    if args.d2:
        d2 = int(args.d2)
    else:
        d2 = 50
    if args.d3:
        d3 = int(args.d3)
    else:
        d3 = 50
    if args.N:
        N = int(args.N)
    else:
        N = 2500
    if args.T:
        T = int(args.T)
    else:
        T = 100
    if args.r:
        r = int(args.r)
    else:
        r = 10
    if args.sigma:
        sigma = float(args.sigma)
    else:
        sigma = 1.0

    # Generate synthetic data
    X, Y, Z, A, R, task_function, cov_X, true_B = generate_synthetic_data(d1, d2, d3, N, T, r, sigma)
    #X, Y, Z, A, R, task_function, cov_X, cov_X_list, true_B = generate_synthetic_data(d1, d2, d3, N, T, r, sigma)

    # Pickle the data to run tensor regression on
    pickle.dump(X, open("synthetic_data/X.pkl", "wb"))
    pickle.dump(Y, open("synthetic_data/Y.pkl", "wb"))
    pickle.dump(Z, open("synthetic_data/Z.pkl", "wb"))
    pickle.dump(A, open("synthetic_data/A.pkl", "wb"))
    pickle.dump(R, open("synthetic_data/R.pkl", "wb"))
    pickle.dump(task_function, open("synthetic_data/task_function.pkl", "wb"))
    pickle.dump(cov_X, open("synthetic_data/cov_X.pkl", "wb"))
    # pickle.dump(cov_X_list, open("synthetic_data/cov_X_list.pkl", "wb"))
    pickle.dump(true_B, open("synthetic_data/true_B.pkl", "wb"))
    end = time.time()
    print("Time to generate data: {}".format(end - start))
