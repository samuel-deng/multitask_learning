import numpy as np
import argparse
import pickle
from generate_data import generate_synthetic_data
from tensor_regression import generate_covariate_X, batch_grad_descent
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.tenalg import inner
from tensorly.cp_tensor import cp_to_tensor
from numpy.linalg import svd
import matplotlib.pyplot as plt
from numpy.linalg import norm

def schatten1_norm(B):
    total_norm = 0
    for mode in range(B.ndim):
        total_norm += np.linalg.norm(tl.unfold(B, mode), 'nuc')

    total_norm = (1/B.ndim) * total_norm
    return total_norm

if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", help="First dimension (d1).")
    parser.add_argument("--d2", help="Second dimension (d2).")
    parser.add_argument("--d3", help="Third dimension (d3).")
    parser.add_argument("--N", help= "Number of users/examples (N).")
    parser.add_argument("--T", help="Number of tasks (T).")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
    parser.add_argument("--sigma", help="Std. dev. of the noise.")
    parser.add_argument("--iters", help="Number of iterations for grad. desc.")
    parser.add_argument('--load_data', dest='load_data', action='store_true')
    parser.set_defaults(load_data=False)

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
        d3 = 10
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
        sigma = 0.1
    if args.iters:
        iterations = int(args.iters)
    else:
        iterations = 200

    load_data = args.load_data

    # D1 for the lambda parameter
    D1 = np.sqrt(d1) + np.sqrt(d2) + np.sqrt(T) + np.sqrt(d1 * T) + np.sqrt(d2 * T) + np.sqrt(d1 * d2)

    # Generate dataset
    if (load_data):
        A = pickle.load(open("synthetic_data/A.pkl", "rb"))
        X = pickle.load(open("synthetic_data/X.pkl", "rb"))
        R = pickle.load(open("synthetic_data/R.pkl", "rb"))
        Y = pickle.load(open("synthetic_data/Y.pkl", "rb"))
        Z = pickle.load(open("synthetic_data/Z.pkl", "rb"))
        true_B = pickle.load(open("synthetic_data/true_B.pkl", "rb")) # A(I, I, Z), the true value B needs to estimate
        task_function = pickle.load(open("synthetic_data/task_function.pkl", "rb"))
        cov_X = pickle.load(open("synthetic_data/cov_X.pkl", "rb"))
    else:
        X, Y, Z, A, R, task_function, cov_X, true_B = generate_synthetic_data(d1, d2, d3, N, T, r, sigma)

    # PREFIX: Setup prefix for the directory where we save the resultant data
    DIR_PREFIX = "result_data/T_{}/".format(T)
    print("Original A shape: {}".format(A.shape))
    pickle.dump(A, open(DIR_PREFIX + "A_T{}.pkl".format(T), "wb"))
    print("Original Z shape: {}".format(Z.shape))
    pickle.dump(Z, open(DIR_PREFIX + "Z_T{}.pkl".format(T), "wb"))

    # Step 1: Tensor regression
    eta = 0.1
    eps = 0.01
    lambd = (40 * sigma * D1)/np.sqrt(N)
    print("lambda hyperparam = {}".format(lambd))
    B = batch_grad_descent(true_B, A, R, X, Y, cov_X, T, eta, eps, lambd, task_function, iterations=100)
    pickle.dump(B, open(DIR_PREFIX + "B_T{}.pkl".format(T), "wb"))
    B = pickle.load(open(DIR_PREFIX + "B_T{}.pkl".format(T), "rb"))
    print("Norm for true B: {}".format(tl.norm(true_B)))
    print("Distance from true B: {}".format(tl.norm(B - true_B)))
    #random_B = np.random.randn(X.shape[1], Y.shape[1], T)
    #print("Distance from random B: {}".format(tl.norm(random_B - true_B)/tl.norm(true_B)))

    # Step 2: Tensor decomposition
    weights, factors = parafac(B, r)
    B_1 = factors[0]
    B_2 = factors[1]
    B_3 = factors[2]

    # Step 3: SVD of B_3
    U, D, V_T = svd(B_3, full_matrices=False)

    # Step 4: Extract A and Z
    est_Z = U @ np.diag(D)
    print("Estimated Z shape: {}".format(est_Z.shape))
    pickle.dump(est_Z, open(DIR_PREFIX + "Z_hat_T{}.pkl".format(T), "wb"))
    factors[2] = V_T
    est_A = cp_to_tensor((weights, factors))
    print("Estimated A shape: {}".format(est_A.shape))
    pickle.dump(est_A, open(DIR_PREFIX + "A_hat_T{}.pkl".format(T), "wb"))
    print("Distance from true A: {}".format(tl.norm(est_A - A)/tl.norm(A)))