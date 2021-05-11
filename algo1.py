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
        #take top-d3 SVD
    else:
        #concat_mat = np.concatenate((np.identity(r), np.zeros((r, d3 - r))), axis=1)
        #B_3_modified = B_3 @ concat_mat
        U, D, V_T = svd(B_3, full_matrices=False)
        U = U[:, :r]    # top-r SVD
        D = D[:r]
        V_T = V_T[:r, :]
        V_T_modified = np.concatenate((V_T.T, np.zeros((d3-r,r))),axis=0)
        V_T = V_T_modified.T

    # Step 4: Extract A
    factors[2] = V_T.T
    est_A = cp_to_tensor((weights, factors))
    return B, est_A
