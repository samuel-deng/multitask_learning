import numpy as np
import pickle
from generate_data import generate_synthetic_data
from tensor_regression import generate_covariate_X, grad_descent, batch_grad_descent
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.tenalg import inner
from numpy.linalg import svd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Dataset parameters
    d1 = 100
    d2 = 100
    d3 = 100
    N = 2500
    T = 100
    r = 3
    sigma = 0.1

    # D1 for the lambda parameter
    D1 = np.sqrt(d1) + np.sqrt(d2) + np.sqrt(T) + np.sqrt(d1 * T) + np.sqrt(d2 * T) + np.sqrt(d1 * d2)

    # Generate dataset
    # X, Y, Z, A, R, task_function = generate_synthetic_data(d1, d2, d3, N, T, r)

    # Load already generate data
    A = pickle.load(open("synthetic_data/A.pkl", "rb"))
    X = pickle.load(open("synthetic_data/X.pkl", "rb"))
    R = pickle.load(open("synthetic_data/R.pkl", "rb"))
    Y = pickle.load(open("synthetic_data/Y.pkl", "rb"))
    Z = pickle.load(open("synthetic_data/Z.pkl", "rb"))
    true_B = pickle.load(open("synthetic_data/true_B.pkl", "rb")) # A(I, I, Z), the true value B needs to estimate
    task_function = pickle.load(open("synthetic_data/task_function.pkl", "rb"))
    cov_X_list = pickle.load(open("synthetic_data/cov_X_list.pkl", "rb"))

    # Step 1: Tensor regression
    eta = 0.1
    eps = 0.1
    iterations = 50
    lambd = (40 * sigma * D1)/np.sqrt(N)
    print("lambda hyperparam = {}".format(lambd))
    B, error_list = batch_grad_descent(A, R, X, Y, cov_X_list, T, eta, eps, lambd, task_function, iterations)
    pickle.dump(B, open("B.pkl", "wb"))
    pickle.dump(error_list, open("errors.pkl", "wb"))
    B = pickle.load(open("B.pkl", "rb"))
    print("Distance from true B: {}".format(tl.norm(B - true_B)))
    print(tl.norm(B - true_B)/tl.norm(true_B))
    #print("Frobenius Norm from true B: {}".format(np.sqrt(inner((B - true_B), (B - true_B))) / np.sqrt(inner(true_B, true_B)) )
    random_B = np.random.randn(X.shape[1], Y.shape[1], T)
    #print("Frobenius Norm (random B): {}".format(np.sqrt(inner((random_B - true_B), (random_B - true_B))) / np.sqrt(inner(true_B, true_B)))
    print(tl.norm(true_B - random_B) / tl.norm(true_B))

    # Plot tensor regression loss over iters
    #plt.plot(error_list)
    #plt.title("Tensor Regression Cost")
    #plt.xlabel("Number of iterations")
    #plt.ylabel("Cost")
    #plt.show()

    # Step 2: Tensor decomposition
    weights, factors = parafac(B, r)
    B_1 = factors[0]
    B_2 = factors[1]
    B_3 = factors[2]
    print(weights)
    print(B_1.shape)
    print(B_2.shape)
    print(B_3.shape)

    # Step 3: SVD of B_3
    U, D, V_T = svd(B_3, full_matrices=False)

    # Step 4: Extract A and Z
    Z = U @ np.diag(D)    
