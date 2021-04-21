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
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg import mode_dot
from tensorly.tenalg import khatri_rao


def LR(test_data):

    X, y = test_data
    beta_LR = la.pinv((X.T @ X),rcond=0.005) @ X.T @ y

    return beta_LR

def eigs(M):

    eigenValues, eigenVectors = la.eig(M)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    return eigenValues, eigenVectors



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

def MoM(train_data):

    T = len(train_data)
    d = train_data[0][0].shape[1]

    total_n=0
    M = np.zeros(shape=(d, d))
    for i in range(T):
        data = train_data[i]
        X, y = data
        num = y.shape[0]
        total_n += num
        scaled_X = (X.T * y).T
        M += (scaled_X).T @ scaled_X
    M = 1/float(total_n) * M

    return M

def rPCA(M, r):

    B,D,B1 = la.svd(M, full_matrices=False)
    return B[:, :r]
    #eigVals, eigVecs = eigs(M)

    #return eigVecs[:, :r], eigVecs[:, r:]

def MetaLR_w_MOM(train_data, r ):

    T = len(train_data)
    d = train_data[0][0].shape[1]

    M_est = MoM(train_data)
    B1  = rPCA(M_est, r)

    #X,y = test_data
    #X_low = X @ B1
    #alpha_LR = LR((X_low, y))
    #beta_LR = B1 @ alpha_LR

    return B1

def MetaLR_w_FO(train_data, r):

    T = len(train_data)
    n, d = train_data[0][0].shape
    m = T*n

    ms_gradients = grad(MS_Loss)

    B_init = np.random.normal(size=(d,r)).flatten()
    V_init = np.random.normal(size=(T,r)).flatten()
    w = np.concatenate((B_init, V_init))

    res_ms = scipy.optimize.minimize(MS_Loss, w, jac=ms_gradients, method='L-BFGS-B', args=(train_data, d, r, m), options = {'maxiter' : 1000})
    B_gd, V_gd = change_shape(res_ms.x, d, r, T)
    B1 = get_col_space(B_gd, r)

    return B1

# Convert our tensor-based data to the format for MTL
def gen_train_model(cov_X, R, task_function,T):
    N = cov_X.shape[0]

    train_data=[]
    for t in range(T):
        #find entries with task id t
        indices = [i for i, x in enumerate(task_function) if x == t]
        Xt = cov_X[indices,]
        r = np.take(R,indices)
        train_data.append((Xt, r))

    return train_data

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
    parser.add_argument("--sigma", help="Std. dev. of the noise.")
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
    parser.add_argument('--A_and_task_dir', help="Path for the underlying tensor A and Y0, Z0.")
    parser.add_argument('--num_trials', help="Number of trials.")
    parser.add_argument('--method', help="Method (Type of matrix model).")

    # Parse args (otherwise set defaults)
    args = parser.parse_args()
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
        seed = args.seed
    else:
        seed = 42
    if args.num_trials:
        num_trials = args.num_trials
    else:
        num_trials = 20
    if args.method:
        method = args.method
    else:
        method = 'I'

    # Load up the underlying tensor A and Y0 and Z0
    A = pickle.load(open(A_and_task_dir + "A.pkl", "rb"))
    d1, d2, d3 = A.shape
    X = pickle.load(open(A_and_task_dir + "X.pkl", "rb"))
    Y = pickle.load(open(A_and_task_dir + "Y.pkl", "rb"))
    T = Y.shape[0]
    R = pickle.load(open(A_and_task_dir + "R.pkl", "rb"))
    task_function = pickle.load(open(A_and_task_dir + "task_function.pkl", "rb"))

    R_ti  = [R[i, task_function[i]] for i in range(len(task_function))]
    # Need X and Y combined into cov_X to generate training data
    Y_ti = Y[task_function]
    if method == 'I': #dimension of recovered matrix is d1d2xd3
        cov_X = np.einsum('bi,bo->bio', X, Y_ti)
        #cov_X = np.concatenate((X, Y_ti), axis=1)
        cov_X = unfold(cov_X,0)

        # Now we can generate the training data
        # In our paper, alphas are equivalent to the Z vectors in d3
        train_data = gen_train_model(cov_X, R_ti, task_function,T)
        # get B using MoM method from TJJ
        B = MetaLR_w_MOM(train_data, d3 )
        #B = MetaLR_w_FO(train_data, d3 )
    elif method == 'II': #dimension of recovered matrix is (d1+d2)xd3
        cov_X = np.concatenate((X,Y_ti), axis=1)
        train_data = gen_train_model(cov_X, R_ti, task_function,T)
        B = MetaLR_w_MOM(train_data, d3 )
    elif method == 'III': #dimension of recovered matrix is d1x(d2+d3) [needs d1 >= d2+d3]
        cov_X = X
        train_data = gen_train_model(cov_X, R_ti, task_function,T)
        B = MetaLR_w_MOM(train_data, d2+d3 )

    print(B.shape)
    print(np.iscomplexobj(B))

    mse_all = np.zeros((num_trials,10))

    for trial in range(num_trials):
        for N2 in range(20,220,20):
            #load data
            X2 = pickle.load(open(output_dir + 'X2_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))
            Y0 = pickle.load(open(output_dir + 'Y0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))
            Z0 = pickle.load(open(output_dir + 'Z0_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))
            R0 = pickle.load(open(output_dir + 'R_N2_{N2F}_trial_{trialF}.pkl'.format(N2F=N2,trialF=trial),'rb'))

            if method == 'I':
                cov_X0 = np.einsum('bo,i->boi', X2, Y0)
                cov_X0 = unfold(cov_X0, 0)
            elif method == 'II':
                cov_X0 = np.concatenate((X2,np.tile(Y0, (N2,1))), axis=1)
            elif method == 'III':
                cov_X0 = X2
            #get estimate of Z0

            X_low = cov_X0 @ B
            alpha_LR = LR((X_low, R0))
            beta_LR = B @ alpha_LR

            # Now, generate 500 test instances to compare MSE (done in notebook when plotting)
            # Generate user feature vectors X
            user_mu = 0
            user_sigma = 1/np.sqrt(d1)
            X_test = user_sigma * np.random.randn(500, d1) + user_mu # From N(0, 1/sqrt(d1))

            # Find avg. error over all X
            true_R = multi_mode_dot(mode_dot(A, X_test, mode=0), [Y0, Z0], modes=[1,2])
            #estimate with beta_LR
            if method == 'I':
                cov_Xtest = np.einsum('bo,i->boi', X_test, Y0)
                cov_Xtest = unfold(cov_Xtest, 0)
            elif method =='II':
                cov_Xtest = np.concatenate((X_test,np.tile(Y0, (X_test.shape[0], 1))), axis=1)
            elif method == 'III':
                cov_Xtest = X_test
            est_R = cov_Xtest @ beta_LR
            MSE = np.sum(np.square(true_R - est_R))
            MSE = MSE / X_test.shape[0]
            #print('N2:{N2F} trial:{trialF} MSE:{msef}'.format(N2F=N2,trialF=trial,msef=MSE))
            mse_all[trial][(N2-20)//20] = MSE

    avgerr = np.mean(mse_all, axis=0)
    stderr = np.std(mse_all, axis=0)/num_trials

    print('(' + str(avgerr[0]) , end = '')
    for i in range(1,10):
        print(", " + str(avgerr[i]), end='')
    print(')')

    print('(' + str(stderr[0]) , end = '')
    for i in range(1,10):
        print(", " + str(stderr[i]), end='')
    print(')')



