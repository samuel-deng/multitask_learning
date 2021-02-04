import autograd.numpy as np
from autograd import grad
import numpy.linalg as la
import scipy.optimize
import random
import math
import argparse
import time
import pickle
from tensorly import unfold
from generate_data import generate_training_data

def change_shape(w, d, r, T):
    b=w[:d*r]
    v=w[d*r:]
    
    B = np.reshape(b, (d,r))
    V = np.reshape(v, (T,r))
    
    return B, V

def get_col_space(B, r):
    u, _, _ = la.svd(B)
    
    return u[:, 0:r]

def LR_Loss(weights, test_data):
    X = test_data[0]
    y = test_data[1]
    
    n = y.shape[0]
    loss = 1/(2*n)*np.linalg.norm(y-X @ weights)**2
    
    return loss

def MS_Loss(weights, train_data, d, r, m):
    T = len(train_data)
    
    b=weights[:d*r]
    v=weights[d*r:]
    
    B = np.reshape(b, (d,r))
    V = np.reshape(v, (T,r))
    
    loss=0
    for t in range(T):
        X, y = train_data[t]
        loss += 1/(2*m)*np.linalg.norm(y-X @ B @ V[t, :])**2
       
    loss += 1/8*np.linalg.norm(B.T @ B - V.T @ V, "fro")**2
    
    return loss

# Method from Provable Meta-Learning paper
def MetaLR_w_FO(train_data, r, test_data):
    T = len(train_data)
    n, d = train_data[0][0].shape
    m = T*n
    
    ms_gradients = grad(MS_Loss)
    
    B_init = np.random.normal(size=(d,r)).flatten()
    V_init = np.random.normal(size=(T,r)).flatten()
    w = np.concatenate((B_init, V_init))
    
    res_ms = scipy.optimize.minimize(MS_Loss, w, jac=ms_gradients, method='L-BFGS-B', args=(train_data, d, r, m), options = {'maxiter' : 1000})   
    print(res_ms.x.shape)
    B_gd, V_gd = change_shape(res_ms.x, d, r, T)
    B1 = get_col_space(B_gd, r)
    
    X, y = test_data
    X_low = X @ B1
    test_data_new = [X_low, y]
    
    test_gradients = grad(LR_Loss)
    
    w = np.random.normal(size=r).flatten()
    res_test = scipy.optimize.minimize(LR_Loss, w, jac=test_gradients, method='L-BFGS-B', args=(test_data_new), options = {'maxiter' : 1000})  
    alpha_LR = res_test.x
    
    beta_LR = B1 @ alpha_LR
    
    return B_gd, B1, beta_LR

# Convert our tensor-based data to the format for MTL
def gen_train_model(A, cov_X, Z):
    # Convert A to the d1d2 x d3 B tensor
    B = unfold(A, 2).T
    cov_X = unfold(cov_X, 0)
    train_alphas = Z
    N = cov_X.shape[0]
    
    train_data=[]
    for i in range(T):
        r = cov_X @ B @ train_alphas[i] + np.random.normal(size=N)
        train_data.append((X, r))
        
    return train_data, B, train_alphas

def gen_test_model(B, d1, Y0, Z0, N2):
    # Convert Z0 to alpha 
    alpha = Z0 

    # Load up N2 new X instances for covariates
    user_mu = 0
    user_sigma = 1/np.sqrt(d1)
    X = user_sigma * np.random.randn(N2, d1) + user_mu # From N(0, 1/sqrt(d1))

    # Convert X, Y0 into covariates N2 many d1d2 covariates
    cov_X = np.einsum('ij,k->ikj', X, Y0).reshape(X.shape[0], -1) 
    y = cov_X @ B @ alpha + np.random.normal(size=N2)
    return (X, y), alpha

if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", help="First dimension (d1).")
    parser.add_argument("--d2", help="Second dimension (d2).")
    parser.add_argument("--d3", help="Third dimension (d3).")
    parser.add_argument("--N", help= "Number of users/examples (N).")
    parser.add_argument("--N2", help="Number of test users (N2).")
    parser.add_argument("--T", help="Number of tasks (T).")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
    parser.add_argument("--sigma", help="Std. dev. of the noise.")
    parser.add_argument("--iters", help="Number of iterations for grad. desc.")
    parser.add_argument("--lambd", help="Value of hyperparameter lambda.")
    parser.add_argument('--eta', help="Eta (learning rate) parameter.")
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
    parser.add_argument('--A_and_task_dir', help="Path for the underlying tensor A and Y0, Z0.")

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
    if args.N2:
        N2 = int(args.N2)
    else:
        N2 = 50
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
    if args.A_and_task_dir:
        A_and_task_dir = args.A_and_task_dir
    else:
        A_and_task_dir = "./persistent/"
    if args.seed:
        seed = args.seed
    else:
        seed = 42

    # Generate data for training first
    X, Y, Z = generate_training_data(d1, d2, d3, N, T, seed)

    # Load up the underlying tensor A and Y0 and Z0
    A = pickle.load(open(A_and_task_dir + "A.pkl", "rb"))
    Y0 = pickle.load(open(A_and_task_dir + "Y0.pkl", "rb"))
    Z0 = pickle.load(open(A_and_task_dir + "Z0.pkl", "rb"))

    # Need X and Y combined into cov_X to generate training data 
    task_function = np.random.randint(0, T, size=N)
    Y_ti = Y[task_function]
    cov_X = np.einsum('bi,bo->bio', X, Y_ti)

    # Now we can generate the training data
    # In our paper, alphas are equivalent to the Z vectors in d3
    train_data, B, train_alphas = gen_train_model(A, cov_X, Z) 
    print(B.shape)

    # After loading up N2 new X covariates, we can generate the meta-test data 
    test_data, alpha_test = gen_test_model(B, d1, Y0, Z0, N2) 

    # Finally, apply the MTL method
    B_gd, B_meta_fo, beta_meta_fo = MetaLR_w_FO(train_data, r, test_data)
    print(B_gd.shape)
