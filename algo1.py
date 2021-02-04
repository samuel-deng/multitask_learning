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

def algo1(true_B, A, R, X, Y, Z, cov_X, T, eta, eps, r, lambd, task_function, iterations):
    # Save the original A and Z tensors
    # Step 1: Tensor regression
    print("lambda hyperparam = {}".format(lambd))
    B = batch_grad_descent(true_B, A, R, X, Y, Z, cov_X, T, eta, eps, r, lambd, task_function, iterations)

    # Step 2: Tensor decomposition
    weights, factors = parafac(B, r)
    print(len(factors))
    B_1 = factors[0]
    B_2 = factors[1]
    B_3 = factors[2]

    weights, factors = parafac(true_B, r)
    true_B_1 = factors[0]
    true_B_2 = factors[1]
    true_B_3 = factors[2]

    # Step 3: SVD of B_3
    d3 = Z.shape[1]
    if r > d3:
        U, D, V_T = svd(B_3, full_matrices=False)
    else:
        concat_mat = np.concatenate((np.identity(r), np.zeros((r, d3 - r))), axis=1)
        B_3_modified = B_3 @ concat_mat
        U, D, V_T = svd(B_3_modified, full_matrices=False)
        U = U[:, :r]    # top-r SVD
        D = D[:r]
        V_T = V_T[:r, :] 

    # Step 4: Extract A
    factors[2] = V_T.T
    est_A = cp_to_tensor((weights, factors))
    return B, est_A 

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
    parser.add_argument("--lambd", help="Value of hyperparameter lambda.")
    parser.add_argument('--load_data', dest='load_data', action='store_true')
    parser.add_argument('--eta', help="Eta (learning rate) parameter.")
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")
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
        d3 = 50
    if args.N:
        N = int(args.N)
    else:
        N = 5000
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
    if args.iters:
        iterations = int(args.iters)
    else:
        iterations = 200
    if args.lambd:
        lambd = float(args.lambd)
    else:
        lambd = 0.01
    if args.eta:
        eta = float(args.eta)
    else:
        eta = 0.1
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "./"
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
        # cov_X_list = pickle.load(open("synthetic_data/cov_X_list.pkl", "rb"))
    else:
        # X, Y, Z, A, R, task_function, cov_X, cov_X_list, true_B = generate_synthetic_data(d1, d2, d3, N, T, r, sigma)
        X, Y, Z, A, R, task_function, cov_X, true_B = generate_synthetic_data(d1, d2, d3, N, T, r, sigma)

    # Run algorithm 1 (saves everything to .pkl within the function)
    eps = 0.01
    B, est_A = algo1(true_B, A, R, X, Y, Z, cov_X, T, eta, eps, r, lambd, task_function, iterations)

    # Save the results from algorithm 1 to pickle
    print("Original A shape: {}".format(A.shape))
    pickle.dump(A, open(output_dir + "A_T{}.pkl".format(T), "wb"))
    print("Original Z shape: {}".format(Z.shape))
    pickle.dump(Z, open(output_dir + "Z_T{}.pkl".format(T), "wb"))
    print("Original B shape: {}".format(true_B.shape))
    pickle.dump(true_B, open(output_dir + "true_B_T{}.pkl".format(T), "wb"))
    print("Estimated B shape: {}".format(B.shape))
    pickle.dump(B, open(output_dir + "B_T{}.pkl".format(T), "wb"))
    print("Norm for recovered B: {}".format(tl.norm(B)))
    print("Norm for true B: {}".format(tl.norm(true_B)))
    print("Distance from true B: {}".format(tl.norm(B - true_B)))
    print("Estimated A shape: {}".format(est_A.shape))
    pickle.dump(est_A, open(output_dir + "A_hat_T{}.pkl".format(T), "wb"))
    print("Distance from true A: {}".format(tl.norm(est_A - A)/tl.norm(A)))

#    # PREFIX: Setup prefix for the directory where we save the resulting data
#    print("Original A shape: {}".format(A.shape))
#    pickle.dump(A, open(output_dir + "A_T{}.pkl".format(T), "wb"))
#    print("Original Z shape: {}".format(Z.shape))
#    pickle.dump(Z, open(output_dir + "Z_T{}.pkl".format(T), "wb"))
#
#    # Step 1: Tensor regression
#    eps = 0.01
#    print("lambda hyperparam = {}".format(lambd))
#    # B = batch_grad_descent(true_B, A, R, X, Y, cov_X, cov_X_list, T, eta, eps, lambd, task_function, iterations)
#    B = batch_grad_descent(true_B, A, R, X, Y, Z, cov_X, T, eta, eps, r, lambd, task_function, iterations)
#    pickle.dump(B, open(output_dir + "B_T{}.pkl".format(T), "wb"))
#    pickle.dump(true_B, open(output_dir + "true_B_T{}.pkl".format(T), "wb"))
#    print("Norm for recovered B: {}".format(tl.norm(B)))
#    print("Norm for true B: {}".format(tl.norm(true_B)))
#    print("Distance from true B: {}".format(tl.norm(B - true_B)))
#
#    # Step 2: Tensor decomposition
#    weights, factors = parafac(B, r)
#    B_1 = factors[0]
#    B_2 = factors[1]
#    B_3 = factors[2]
#
#    weights, factors = parafac(true_B, r)
#    true_B_1 = factors[0]
#    true_B_2 = factors[1]
#    true_B_3 = factors[2]
#    print("True B_3 shape: {}".format(true_B_3.shape))
#
#    print("Distance for B_1: {}".format(tl.norm(B_1 - true_B_1)))
#    print("Distance for B_2: {}".format(tl.norm(B_2 - true_B_2)))
#
#    # Step 3: SVD of B_3
#    if r > d3:
#        U, D, V_T = svd(B_3, full_matrices=False)
#    else:
#        concat_mat = np.concatenate((np.identity(r), np.zeros((r, d3 - r))), axis=1)
#        B_3_modified = B_3 @ concat_mat
#        U, D, V_T = svd(B_3_modified, full_matrices=False)
#        print("U shape: {}".format(U.shape))
#        print("D shape: {}".format(D.shape))
#        print("V_T shape: {}".format(V_T.shape))
#        U = U[:, :r]    # top-r SVD
#        D = D[:r]
#        V_T = V_T[:r, :] 
#
#    # Step 4: Extract A and Z
#    est_Z = U @ np.diag(D)
#    print("Estimated Z shape: {}".format(est_Z.shape))
#    pickle.dump(est_Z, open(output_dir + "Z_hat_T{}.pkl".format(T), "wb"))
#    factors[2] = V_T.T
#    est_A = cp_to_tensor((weights, factors))
#    print("Estimated A shape: {}".format(est_A.shape))
#    pickle.dump(est_A, open(output_dir + "A_hat_T{}.pkl".format(T), "wb"))
#    print("Distance from true A: {}".format(tl.norm(est_A - A)/tl.norm(A)))
