import numpy as np
import argparse
import pickle
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg import mode_dot
from tensorly.tenalg import khatri_rao
from algo1 import algo1
from algo2 import algo2
from generate_data import generate_training_data
from generate_data import generate_responses
from generate_data import generate_A_tensor

def least_squares(A1, A2, X, Y0, R, r):
    # Get the CP decomposition of A
    #W, factors = parafac(A, r, normalize_factors=True)
    #print(W)
    #A1 = factors[0]
    #A2 = factors[1]
    #A3 = factors[2]

    # Construct \hat{V}
    Y_prod = Y0.T @ A2
    Y_prod = np.reshape(Y_prod, (Y_prod.shape[0], 1)).T
    X_prod = X @ A1
    kr_prod = khatri_rao([Y_prod, X_prod])
    #V = kr_prod @ np.diag(W)

    #inverse_term = (A3 @ V.T) @ (V @ A3.T)
    wt = np.linalg.pinv(kr_prod) @ R
    # Z = np.linalg.pinv(inverse_term) @ (A3 @ V.T @ R)
    return wt

def generate_test_data(A, Y, Z, N2):
    d2, T = Y.shape
    d1, d2, d3 = A.shape
    #get new Y2 and Z2 by randomly averaging over a few tasks
    tid = np.random.randint(0,T,size=50)
    Ysi = Y[tid]
    Zsi = Z[tid]
    Y2 = np.mean(Ysi, axis=0)
    Z2 = np.mean(Zsi, axis=0)

    user_mu = 0
    user_sigma = 1/np.sqrt(d1)
    X2 = user_sigma * np.random.randn(N2, d1) + user_mu # From N(0, 1/sqrt(d1))

    noise = np.random.normal(0, 1, N2)
    R2 = [multi_mode_dot(A, [X2[u], Y2, Z2], modes=[0,1,2]) for u in range(N2)]
    R2 = np.asarray(R2) + noise

    return X2, Y2, Z2, R2

def generate_new_data(A, Y, Z, T, samples):
    N = samples * T #10 samples per task on average
    d1, d2, d3 = A.shape
    # Generate user feature vectors X
    user_mu = 0
    user_sigma = 1/np.sqrt(d1)
    X1 = user_sigma * np.random.randn(N, d1) + user_mu # From N(0, 1/sqrt(d1))

    #sample T tasks u.a.r. from (Y,Z)
    tasks_id = np.random.randint(0, 200, size=T)
    Y1 = Y[tasks_id]
    Z1 = Z[tasks_id]
    # Generate the responses
    task_function = np.random.randint(0, T, size=N)
    Y1_ti = Y1[task_function]
    Z1_ti = Z1[task_function]
    R = [multi_mode_dot(A, [X1[u], Y1_ti[u], Z1_ti[u]], modes=[0,1,2]) for u in range(N)]
    R = np.asarray(R)
    print(R.shape)
    noise = np.random.normal(0, 1, N)
    R = np.asarray(R) + noise
    return X1, Y1, Z1, R, task_function


if __name__ == '__main__':
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", help="Std. dev. of the noise.")
    parser.add_argument('--output_dir', help="Path to output the generated tensors to.")
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
    parser.add_argument('--A_and_task_dir', help="Path for the underlying tensor A and Y0, Z0.")
    parser.add_argument('--num_trials', help="Number of Trials.")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
    parser.add_argument("--iters", help="Number of iterations for grad. desc.")
    parser.add_argument("--lambd", help="Value of hyperparameter lambda.")
    parser.add_argument('--eta', help="Eta (learning rate) parameter.")
    parser.add_argument('--trial', help="current trial.")
    parser.add_argument('--load_data', help="Load / Generate Data (default is load (1)).")
    parser.add_argument('--method', help="algo1 (1) or algo2 (2)")


    # Parse args (otherwise set defaults)
    args = parser.parse_args()
    if args.method:
        method = int(args.method)
    else:
        method = 1
    if args.load_data:
        load_data = int(args.load_data)
    else:
        load_data = 1
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
    if args.num_trials:
        num_trials = int(args.num_trials)
    else:
        num_trials = 20
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
    if args.trial:
        trial = int(args.trial)
    else:
        trial = 0

    # Load A and est_A from saved directory
    A = pickle.load(open(A_and_task_dir + "A.pkl", "rb"))
    #est_A = pickle.load(open(A_and_task_dir + "est_A.pkl", "rb"))
    d1, d2, d3 = A.shape
    # Load Y and Z from saved directory (contains 200 tasks)
    Y = pickle.load(open(A_and_task_dir + 'Y.pkl', 'rb'))
    Z = pickle.load(open(A_and_task_dir + 'Z.pkl', 'rb'))

    if load_data == 0:
        #generate test data (N2=50 samples)
        X2, Y2, Z2, R2 = generate_test_data(A, Y, Z, 50)
        pickle.dump(X2, open(output_dir + 'X2_test.pkl', 'wb'))
        pickle.dump(Y2, open(output_dir + 'Y2_test.pkl', 'wb'))
        pickle.dump(Z2, open(output_dir + 'Z2_test.pkl', 'wb'))
        pickle.dump(R2, open(output_dir + 'R2_test.pkl', 'wb'))
    else:
        #load meta test data
        X2 = pickle.load(open(output_dir + 'X2_test.pkl', 'rb'))
        Y2 = pickle.load(open(output_dir + 'Y2_test.pkl', 'rb'))
        Z2 = pickle.load(open(output_dir + 'Z2_test.pkl', 'rb'))
        R2 = pickle.load(open(output_dir + 'R2_test.pkl', 'rb'))

    mse_all = np.zeros(10)

    T = 50
    for samples in range(50,550,50):
        #generate data
        if load_data == 0:
            X1, Y1, Z1, R1, task_function = generate_new_data(A, Y, Z, T, samples)
            #save data
            pickle.dump(X1, open(output_dir + 'X1_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'wb'))
            pickle.dump(Y1, open(output_dir + 'Y1_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'wb'))
            pickle.dump(Z1, open(output_dir + 'Z1_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'wb'))
            pickle.dump(R1, open(output_dir + 'R1_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'wb'))
            pickle.dump(task_function, open(output_dir + 'task_function_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'wb'))

            print('done ' + str(trial) +' ' + str(samples))
        else:
            #load data
            X1 = pickle.load(open(output_dir + 'X1_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'rb'))
            Y1 = pickle.load(open(output_dir + 'Y1_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'rb'))
            Z1 = pickle.load(open(output_dir + 'Z1_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'rb'))
            R1 = pickle.load(open(output_dir + 'R1_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'rb'))
            task_function = pickle.load(open(output_dir + 'task_function_samples_{TF}_trial_{trialF}.pkl'.format(TF=samples,trialF=trial),'rb'))

        #estimate tensor A
        true_B = mode_dot(A, Z1, mode=2)
        #Perform algorithm 1 to get estimated A
        Y_ti = Y[task_function]
        if method == 1:
            cov_X = np.einsum('bi,bo->bio', X1, Y_ti)
            eps = 0.01
            B, est_A1, est_A2 = algo1(true_B, A, R1, X1, Y1, Z1, cov_X, T, eta, eps, r, lambd, task_function, iterations)
        else:
            N,_ = X1.shape
            _,d3 = Z1.shape
            #Ri = [R1[i][task_function[i]] for i in range(N)]
            est_A1, est_A2 = algo2(R1, X1, Y1, task_function, 10, d3, A)

        pickle.dump(est_A1, open(output_dir + 'est_A1_trial_{trialF}_samples_{TF}_method_{methodF}.pkl'.format(trialF=trial, TF=samples, methodF=method), 'wb'))
        pickle.dump(est_A2, open(output_dir + 'est_A2_trial_{trialF}_samples_{TF}_method_{methodF}.pkl'.format(trialF=trial, TF=samples, methodF=method), 'wb'))

        # Need to get A^3^TZ_0 from A to perform least squares on new task
        est_wt = least_squares(est_A1, est_A2,  X2, Y2, R2, 10)

        # Now, generate 500 test instances to compare MSE (done in notebook when plotting)
        # Generate user feature vectors X
        user_mu = 0
        user_sigma = 1/np.sqrt(d1)
        X_test = user_sigma * np.random.randn(500, d1) + user_mu # From N(0, 1/sqrt(d1))

        # Find avg. error over all X
        Y_prod = Y2.T @ est_A2
        Y_prod = np.reshape(Y_prod, (Y_prod.shape[0], 1)).T
        X_prod = np.matmul(X_test, est_A1)
        kr_prod = khatri_rao([Y_prod, X_prod])
        est_R = kr_prod @ est_wt
        true_R = multi_mode_dot(mode_dot(A, X_test, mode=0), [Y2, Z2], modes=[1,2])
        MSE = np.sum(np.square(true_R - est_R))
        MSE = MSE / X_test.shape[0]
        mse_all[(samples-50)//50] = MSE
        print(MSE)
    print(mse_all)
    pickle.dump(mse_all, open(output_dir + 'mse_trial_{trialF}_samples_{samplesF}.pkl'.format(trialF=trial, samplesF=samples),'wb'))
    print(np.mean(mse_all, axis=0))
    print(np.std(mse_all, axis=0)/num_trials)


