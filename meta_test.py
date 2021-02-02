import numpy as np
import argparse
import pickle


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
    parser.add_argument('--eta', help="Eta (learning rate) parameter.")
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")

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
        N = 2000
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
 
    # Generate synthetic training data
    X, Y, Z, A, R, task_function, cov_X, true_B = generate_synthetic_data(d1, d2, d3, N, T, r, sigma)
    print("Original A shape: {}".format(A.shape))
    pickle.dump(A, open(output_dir + "A_T{}.pkl".format(T), "wb"))
    print("Original Z shape: {}".format(Z.shape))
    pickle.dump(Z, open(output_dir + "Z_T{}.pkl".format(T), "wb"))

    # Perform algorithm 1 to get estimated A
    # Step 1: Tensor regression
    eps = 0.01
    print("lambda hyperparam = {}".format(lambd))
    # B = batch_grad_descent(true_B, A, R, X, Y, cov_X, cov_X_list, T, eta, eps, lambd, task_function, iterations)
    B = batch_grad_descent(true_B, A, R, X, Y, Z, cov_X, T, eta, eps, r, lambd, task_function, iterations)
    pickle.dump(B, open(output_dir + "B_T{}.pkl".format(T), "wb"))
    pickle.dump(true_B, open(output_dir + "true_B_T{}.pkl".format(T), "wb"))
    print("Norm for recovered B: {}".format(tl.norm(B)))
    print("Norm for true B: {}".format(tl.norm(true_B)))
    print("Distance from true B: {}".format(tl.norm(B - true_B)))

    # Step 2: Tensor decomposition
    weights, factors = parafac(B, r)
    B_1 = factors[0]
    B_2 = factors[1]
    B_3 = factors[2]

    weights, factors = parafac(true_B, r)
    true_B_1 = factors[0]
    true_B_2 = factors[1]
    true_B_3 = factors[2]
    print("True B_3 shape: {}".format(true_B_3.shape))

    print("Distance for B_1: {}".format(tl.norm(B_1 - true_B_1)))
    print("Distance for B_2: {}".format(tl.norm(B_2 - true_B_2)))

    # Step 3: SVD of B_3
    if r > d3:
        U, D, V_T = svd(B_3, full_matrices=False)
    else:
        concat_mat = np.concatenate((np.identity(r), np.zeros((r, d3 - r))), axis=1)
        B_3_modified = B_3 @ concat_mat
        U, D, V_T = svd(B_3_modified, full_matrices=False)
        print("U shape: {}".format(U.shape))
        print("D shape: {}".format(D.shape))
        print("V_T shape: {}".format(V_T.shape))
        U = U[:, :r]    # top-r SVD
        D = D[:r]
        V_T = V_T[:r, :] 

    # Step 4: Extract A and Z
    est_Z = U @ np.diag(D)
    print("Estimated Z shape: {}".format(est_Z.shape))
    pickle.dump(est_Z, open(output_dir + "Z_hat_T{}.pkl".format(T), "wb"))
    factors[2] = V_T.T
    est_A = cp_to_tensor((weights, factors))
    print("Estimated A shape: {}".format(est_A.shape))
    pickle.dump(est_A, open(output_dir + "A_hat_T{}.pkl".format(T), "wb"))
    print("Distance from true A: {}".format(tl.norm(est_A - A)/tl.norm(A)))
    
    # Generate a new task Y0, Z0 with N2 new users
    X, Y0, Z0, R = generate_test(d1, d2, d3, N2, A)
    est_Z0 = least_squares(A, X, Y0, W, R)

    # Find avg. error over all X
    err_sum = 0 
    for X_i in range(X.shape[0]):
        R = multi_mode_dot(mode_dot(A, X, mode=0), [Y0, Z0], modes=[1, 2])
        est_R = multi_mode_dot(mode_dot(est_A, X, mode=0), [Y0, est_Z0], modes=[1, 2])
        err_sum += (R - est_R) ** 2
    err_sum = err_sum / X.shape[0]
    print("MSE for N2 = {}: {}".format(N2, err_sum))



