import numpy as np
import pickle
from numpy.linalg import  svd
import scipy.linalg as la
from tensorly.cp_tensor import cp_to_tensor
from tensorly import kr
from tensorly.decomposition import parafac
import tensorly as tl
import math

def algo2(R, X, Y, task_function, r, d3, A):

    N, d1 = X.shape

    #get an estimate of A1
    M = np.zeros((d1,d1))
    for i in range(N):
        M += R[i]*R[i] * (X[i] @ X[i].T)
    M = 1/float(N) * M
    B, D, B1  = svd(M, full_matrices=False)
    A1 = B[:, :r]

    #get an estimate of A2
    _, d2 = Y.shape
    M = np.zeros((d2, d2))
    for i in range(N):
        M += R[i]*R[i] * (Y[task_function[i]] @ Y[task_function[i]].T)
    M = 1/float(N) * M
    B, D, B1 = svd(M, full_matrices=False)
    A2 = B[:, :r]

    #use estimates A1 and A2 to get an estimate of A3
    M = np.zeros((r, r))
    #print(A2.shape)
    #print(A1.shape)
    krprod = kr((A2, A1))
    for i in range(N):
        P = np.kron(Y[task_function[i]], X[i]) @ krprod
        M += R[i]*R[i] * (P @ P.T)
    #print(M.shape)
    B, D, B1 = svd(M, full_matrices=False)
    C3 = B[:, :r]
    C3 = np.concatenate((C3, np.zeros((r, d3-r))), axis=1)
    A3 = C3.T
    #print(A3.shape)
    weights = np.ones(r)


    est_A = cp_to_tensor((weights, [A1, A2, A3]))
    #print(est_A.shape)
    weights, factors = parafac(A,r)
    #theta1 = math.acos(np.linalg.norm(factors[0].T @ A1, ord=2))
    print(np.linalg.norm(factors[0].T @ A1, ord=2))
    print(np.linalg.norm(factors[1].T @ A2, ord=2))
    #print(math.sin(theta1))
    #theta2 = math.acos(np.linalg.norm(factors[1].T @ A2, ord=2))
    #print(math.sin(theta2))
    #print(tl.norm(factors[2]-A3)/ tl.norm(A3))
    return A1, A2

