import numpy as np
import pickle
from generate_data import generate_synthetic_data
from tensor_regression import generate_covariate_X, grad_descent

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
    task_function = pickle.load(open("synthetic_data/task_function.pkl", "rb"))

    # Tensor regression
    eta = 0.1
    eps = 0.1
    lambd = (40 * sigma * D1)/np.sqrt(N)
    print("lambda hyperparam = {}".format(lambd))
    B = grad_descent(A, R, X, Y, T, eta, eps, lambd, task_function) # Works! Gets stuck at around ~0.3 loss

    # Tensor decomposition

