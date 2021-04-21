import numpy as np
import argparse
import pickle
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg import mode_dot
from tensorly.tenalg import khatri_rao
from algo1 import algo1
from generate_data import generate_training_data
from generate_data import generate_responses
from generate_data import generate_A_tensor

def least_squares(A, X, Y0, R, r):
    # Get the CP decomposition of A
    W, factors = parafac(A, r, normalize_factors=True)
    print(W)
    A1 = factors[0]
    A2 = factors[1]
    A3 = factors[2]

    # Construct \hat{V}
    Y_prod = Y0.T @ A2
    Y_prod = np.reshape(Y_prod, (Y_prod.shape[0], 1)).T
    X_prod = X @ A1
    kr_prod = khatri_rao([Y_prod, X_prod])
    V = kr_prod @ np.diag(W)

    # Construct hat_{Z}
    inverse_term = (A3 @ V.T) @ (V @ A3.T)
    Z = np.linalg.pinv((V @ A3.T)) @ R
    # Z = np.linalg.pinv(inverse_term) @ (A3 @ V.T @ R)
    return Z

def generate_new_task(d1, d2, d3, N2, A, seed=42):
    # Seed the randomness
    np.random.seed(seed)

    # Generate user feature vectors X
    user_mu = 0
    user_sigma = 1/np.sqrt(d1)
    X = user_sigma * np.random.randn(N2, d1) + user_mu # From N(0, 1/sqrt(d1))

    # Generate test task (Y, Z)
    Y_mu = 0
    Y_sigma = 1/np.sqrt(d2)
    Z_mu = 0
    Z_sigma = 1/np.sqrt(d3)
    Y0 = Y_sigma * np.random.randn(d2) + Y_mu
    Z0 = Z_sigma * np.random.randn(d3) + Z_mu

    # Generate the responses
    noise = np.random.normal(0, 1, (N2))
    R = multi_mode_dot(mode_dot(A, X, mode=0), [Y0, Z0], modes=[1,2])
    R = np.asarray(R) + noise
    return X, Y0, Z0, R

if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N2", help="Number of test users (N2).")
    parser.add_argument("--sigma", help="Std. dev. of the noise.")
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
    parser.add_argument('--A_and_task_dir', help="Path for the underlying tensor A and Y0, Z0.")

    # Parse args (otherwise set defaults)
    args = parser.parse_args()
    if args.N2:
        N2 = int(args.N2)
    else:
        N2 = 50
    if args.sigma:
        sigma = float(args.sigma)
    else:
        sigma = 1.0
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "./"
    if args.A_and_task_dir:
        A_and_task_dir = args.A_and_task_dir
    else:
        A_and_task_dir = "./persistent/"
    if args.seed:
        seed = int(args.seed)
    else:
        seed = 42

    # Load A and est_A from saved directory
    A = pickle.load(open(A_and_task_dir + "A.pkl", "rb"))
    est_A = pickle.load(open(A_and_task_dir + "est_A.pkl", "rb"))
    d1, d2, d3 = A.shape
    Y0 = (1/np.sqrt(d2)) * np.random.randn(d2)
    Z0 = (1/np.sqrt(d3)) * np.random.randn(d3)

    # Generate N2 instances X to estimate Z0
    user_mu = 0
    user_sigma = 1/np.sqrt(d1)
    X = user_sigma * np.random.randn(N2, d1) + user_mu # From N(0, 1/sqrt(d1))
    # R = generate_responses(A, X, Y0, Z0, T)
    R = multi_mode_dot(mode_dot(A, X, mode=0), [Y0, Z0], modes=[1,2])
    print("X shape: {}".format(X.shape))
    print("Y0 shape: {}".format(Y0.shape))
    print("Z0 shape: {}".format(Z0.shape))

    # Need to get W from A to perform least squares on new task
    est_Z0 = least_squares(est_A, X, Y0, R, r)

    # Now, generate 500 test instances to compare MSE (done in notebook when plotting)
    # Generate user feature vectors X
    user_mu = 0
    user_sigma = 1/np.sqrt(d1)
    X_test = user_sigma * np.random.randn(500, d1) + user_mu # From N(0, 1/sqrt(d1))

    # Find avg. error over all X
    true_R = multi_mode_dot(mode_dot(A, X_test, mode=0), [Y0, Z0], modes=[1,2])
    est_R = multi_mode_dot(mode_dot(est_A, X_test, mode=0), [Y0, est_Z0], modes=[1,2])
    MSE = np.sum(np.square(true_R - est_R))
    MSE = MSE / X.shape[0]

    # Save the necessary tensors and matrices to find MSE again and plot
    # Current usage:
    #     output_dir = 'meta_test_results/trial_<N>/'
    pickle.dump(A, open(output_dir + "A_N2_{}.pkl".format(N2), "wb"))
    pickle.dump(est_A, open(output_dir + "A_hat_N2_{}.pkl".format(N2), "wb"))
    pickle.dump(X, open(output_dir + "X_cal_N2_{}.pkl".format(N2), "wb"))
    pickle.dump(X_test, open(output_dir + "X_test_N2_{}.pkl".format(N2), "wb"))
    pickle.dump(Y0, open(output_dir + "Y0_N2_{}.pkl".format(N2), "wb"))
    pickle.dump(Z0, open(output_dir + "Z0_N2_{}.pkl".format(N2), "wb"))
    pickle.dump(est_Z0, open(output_dir + "Z0_hat_N2_{}.pkl".format(N2), "wb"))
    pickle.dump(true_R, open(output_dir + "true_R_N2_{}.pkl".format(N2), "wb"))
    pickle.dump(est_R, open(output_dir + "est_R_N2_{}.pkl".format(N2), "wb"))

