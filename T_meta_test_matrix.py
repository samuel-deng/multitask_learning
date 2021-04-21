import numpy as np
import argparse
import pickle
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg import mode_dot
from tensorly.tenalg import khatri_rao
from tensorly import unfold
from algo1 import algo1
from generate_data import generate_training_data
from generate_data import generate_responses
from generate_data import generate_A_tensor
from meta_test_matrix import MetaLR_w_MOM
from meta_test_matrix import LR
from meta_test_matrix import gen_train_model

def least_squares(A, X, Y0, R, r):
    # Get the CP decomposition of A
    W, factors = parafac(A, r, normalize_factors=True)
    #print(W)
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


if __name__ == '__main__':
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", help="Std. dev. of the noise.")
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
    parser.add_argument('--A_and_task_dir', help="Path for the underlying tensor A and Y0, Z0.")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
    parser.add_argument("--iters", help="Number of iterations for grad. desc.")
    parser.add_argument("--lambd", help="Value of hyperparameter lambda.")
    parser.add_argument('--eta', help="Eta (learning rate) parameter.")
    #parser.add_argument('--trial', help="current trial.")
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
        output_dir = "./result_data/all_trials_T/"
    if args.A_and_task_dir:
        A_and_task_dir = args.A_and_task_dir
    else:
        A_and_task_dir = "./persistent/"
    if args.seed:
        seed = int(args.seed)
    else:
        seed = 11
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
        iterations = 100
    if args.lambd:
        lambd = float(args.lambd)
    else:
        lambd = 0.01
    if args.eta:
        eta = float(args.eta)
    else:
        eta = 0.1
    #if args.trial:
    #    trial = int(args.trial)
    #else:
    #    trial = 0
    if args.method:
        method = args.method
    else:
        method = 'I'

    # Load A and est_A from saved directory
    A = pickle.load(open(A_and_task_dir + "A.pkl", "rb"))
    #est_A = pickle.load(open(A_and_task_dir + "est_A.pkl", "rb"))
    d1, d2, d3 = A.shape
    # Load Y and Z from saved directory (contains 200 tasks)
    Y = pickle.load(open(A_and_task_dir + 'Y.pkl', 'rb'))
    Z = pickle.load(open(A_and_task_dir + 'Z.pkl', 'rb'))

    #generate test data (N2=50 samples)
    #X2, Y2, Z2, R2 = generate_test_data(A, Y, Z, 50)
    #load meta test data
    X2 = pickle.load(open(output_dir + 'X2_test.pkl', 'rb'))
    Y2 = pickle.load(open(output_dir + 'Y2_test.pkl', 'rb'))
    Z2 = pickle.load(open(output_dir + 'Z2_test.pkl', 'rb'))
    R2 = pickle.load(open(output_dir + 'R2_test.pkl', 'rb'))
    N2 = 50

    num_trials = 20
    mse_all = np.zeros((20,10))
    for trial in range(20):
        print('starting ' + str(trial))
        for T in range(20,220,20):
            #print('starting ' + str(trial) +' ' + str(T))
            #load data
            X1 = pickle.load(open(output_dir + 'X1_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'rb'))
            Y1 = pickle.load(open(output_dir + 'Y1_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'rb'))
            Z1 = pickle.load(open(output_dir + 'Z1_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'rb'))
            R1 = pickle.load(open(output_dir + 'R1_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'rb'))
            task_function = pickle.load(open(output_dir + 'task_function_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'rb'))
            #estimate tensor A
            true_B = mode_dot(A, Z1, mode=2)
            #Perform algorithm 1 to get estimated A
            Y_ti = Y[task_function]

            if method == 'I': #dimension of recovered matrix is d1d2xd3
                cov_X = np.einsum('bi,bo->bio', X1, Y_ti)
                #cov_X = np.concatenate((X, Y_ti), axis=1)
                cov_X = unfold(cov_X,0)

                # Now we can generate the training data
                # In our paper, alphas are equivalent to the Z vectors in d3
                train_data = gen_train_model(cov_X, R1, task_function,T)
                # get B using MoM method from TJJ
                B = MetaLR_w_MOM(train_data, d3 )
                #B = MetaLR_w_FO(train_data, d3 )
            elif method == 'II': #dimension of recovered matrix is (d1+d2)xd3
                cov_X = np.concatenate((X1,Y_ti), axis=1)
                train_data = gen_train_model(cov_X, R1, task_function,T)
                B = MetaLR_w_MOM(train_data, d3 )
            elif method == 'III': #dimension of recovered matrix is d1x(d2+d3) [needs d1 >= d2+d3]
                cov_X = X1
                train_data = gen_train_model(cov_X, R1, task_function,T)
                B = MetaLR_w_MOM(train_data, d2+d3 )

            if method == 'I':
                cov_X0 = np.einsum('bo,i->boi', X2, Y2)
                cov_X0 = unfold(cov_X0, 0)
            elif method == 'II':
                cov_X0 = np.concatenate((X2,np.tile(Y2, (N2,1))), axis=1)
            elif method == 'III':
                cov_X0 = X2
            #get estimate of Z2

            X_low = cov_X0 @ B
            alpha_LR = LR((X_low, R2))
            beta_LR = B @ alpha_LR


            # Now, generate 500 test instances to compare MSE (done in notebook when plotting)
            # Generate user feature vectors X
            user_mu = 0
            user_sigma = 1/np.sqrt(d1)
            X_test = user_sigma * np.random.randn(500, d1) + user_mu # From N(0, 1/sqrt(d1))
            true_R = multi_mode_dot(mode_dot(A, X_test, mode=0), [Y2, Z2], modes=[1,2])

            # Find avg. error over all X
            #estimate with beta_LR
            if method == 'I':
                cov_Xtest = np.einsum('bo,i->boi', X_test, Y2)
                cov_Xtest = unfold(cov_Xtest, 0)
            elif method =='II':
                cov_Xtest = np.concatenate((X_test,np.tile(Y2, (X_test.shape[0], 1))), axis=1)
            elif method == 'III':
                cov_Xtest = X_test
            est_R = cov_Xtest @ beta_LR

            MSE = np.sum(np.square(true_R - est_R))
            MSE = MSE / X_test.shape[0]
            mse_all[trial][(T-20)//20] = MSE
            #print(MSE)
    #pickle.dump(mse_all, open(output_dir + 'mse_matrix_{methodF}_trial_{trialF}.pkl'.format(methodF = method, trialF=trial),'wb'))
    print(np.mean(mse_all, axis=0))
    print(np.std(mse_all, axis=0)/num_trials)


