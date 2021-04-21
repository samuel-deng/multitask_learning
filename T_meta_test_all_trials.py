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

def generate_new_data(A, Y, Z, T):
    N = 20 * T #10 samples per task on average
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

    #generate test data (N2=50 samples)
    X2, Y2, Z2, R2 = generate_test_data(A, Y, Z, 50)
    #load meta test data
    #X2 = pickle.load(open(output_dir + 'X2_test.pkl', 'rb'))
    #Y2 = pickle.load(open(output_dir + 'Y2_test.pkl', 'rb'))
    #Z2 = pickle.load(open(output_dir + 'Z2_test.pkl', 'rb'))
    #R2 = pickle.load(open(output_dir + 'R2_test.pkl', 'rb'))

    mse_all = np.zeros(10)

    for T in range(20,220,20):
        #generate data
        X1, Y1, Z1, R1, task_function = generate_new_data(A, Y, Z, T)
        #save data
        pickle.dump(X1, open(output_dir + 'X1_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'wb'))
        pickle.dump(Y1, open(output_dir + 'Y1_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'wb'))
        pickle.dump(Z1, open(output_dir + 'Z1_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'wb'))
        pickle.dump(R1, open(output_dir + 'R1_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'wb'))
        pickle.dump(task_function, open(output_dir + 'task_function_T_{TF}_trial_{trialF}.pkl'.format(TF=T,trialF=trial),'wb'))

        print('done ' + str(trial) +' ' + str(T))
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
        cov_X = np.einsum('bi,bo->bio', X1, Y_ti)
        eps = 0.01
        B, est_A = algo1(true_B, A, R1, X1, Y1, Z1, cov_X, T, eta, eps, r, lambd, task_function, iterations)

        est_Z2 = least_squares(est_A, X2, Y2, R2, r)

        # Now, generate 500 test instances to compare MSE (done in notebook when plotting)
        # Generate user feature vectors X
        user_mu = 0
        user_sigma = 1/np.sqrt(d1)
        X_test = user_sigma * np.random.randn(500, d1) + user_mu # From N(0, 1/sqrt(d1))

        # Find avg. error over all X
        true_R = multi_mode_dot(mode_dot(A, X_test, mode=0), [Y2, Z2], modes=[1,2])
        est_R = multi_mode_dot(mode_dot(est_A, X_test, mode=0), [Y2, est_Z2], modes=[1,2])
        MSE = np.sum(np.square(true_R - est_R))
        MSE = MSE / X_test.shape[0]
        mse_all[(T-20)//20] = MSE
        print(MSE)
    print(mse_all)
    pickle.dump(mse_all, open(output_dir + 'mse_trial_{trialF}.pkl'.format(trialF=trial),'wb'))
    #print(np.mean(mse_all, axis=0))
    #print(np.std(mse_all, axis=0)/num_trials)


