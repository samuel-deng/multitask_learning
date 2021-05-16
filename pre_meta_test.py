import argparse
import numpy as np
import pickle
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
from tensorly.tenalg import mode_dot
from tensorly.tenalg import khatri_rao
from generate_data import generate_training_data
from generate_data import generate_responses
from generate_data import generate_A_tensor
from algo1 import algo1
from algo2 import algo2
from numpy.linalg import norm
import tensorly as tl

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
    parser.add_argument('--seed', help="Seed for the randomness of data generation.")
<<<<<<< HEAD
    parser.add_argument('--load_data', help="Path for the underlying data.")
    parser.add_argument('--save_data', help="Specify where you would like to save the data.")
    parser.add_argument('--estimated_data', help="Specify where to put the estimates (A_1 and A_2).")
=======
    parser.add_argument('--load_data', help="Load data (1) or generate data (0).")
    parser.add_argument('--A_and_task_dir', help="Path for the underlying tensor A and Y0, Z0.")
>>>>>>> 0d91aeda7e0b1427a993232ef0c716b0d0a03839

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
        N = 1000
    if args.T:
        T = int(args.T)
    else:
        T = 200
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
<<<<<<< HEAD
=======
    if args.A_and_task_dir:
        A_and_task_dir = args.A_and_task_dir
    else:
        A_and_task_dir = "./meta_test_results/persistent/"
>>>>>>> 0d91aeda7e0b1427a993232ef0c716b0d0a03839
    if args.seed:
        seed = args.seed
    else:
        seed = 42
<<<<<<< HEAD
    #if args.A_and_task_dir:
    #    A_and_task_dir = args.A_and_task_dir
    #else:
    #    A_and_task_dir = "meta_test_results/persistent/"
    if args.load_data:
        A = pickle.load(open(args.load_data + "A.pkl", "rb"))
        X = pickle.load(open(args.load_data + "X.pkl", "rb"))
        Y = pickle.load(open(args.load_data + "Y.pkl", "rb"))
        Z = pickle.load(open(args.load_data + "Z.pkl", "rb"))   # NOTE: the contents of Z are never actually used, only shape
        R = pickle.load(open(args.load_data + "R.pkl", "rb"))
        task_function = pickle.load(open(args.load_data + "task_function.pkl", "rb"))
    else:
        print("Generating synthetic data...")
        # Generate A tensor
        A = generate_A_tensor(d1, d2, d3, r)

        # Generate synthetic training data
        X, Y, Z = generate_training_data(d1, d2, d3, N, T)
        noise = np.random.normal(0, 1, (N, T))
        R = generate_responses(A, X, Y, Z, T)
        R += noise
        task_function = np.random.randint(0, T, size=N)
    if args.save_data:
        save_data = args.save_data
    else:
        save_data = False
    if args.estimated_data:
        estimated_data = args.estimated_data
    else:
        estimated_data = "estimated_data/"

    # ALGO 1
    #print("Executing Algorithm 1...")
    #true_B = mode_dot(A, Z, mode=2)
    #Y_ti = Y[task_function]
    #cov_X = np.einsum('bi,bo->bio', X, Y_ti)

    # Perform algorithm 1 to get estimated A
    #eps = 0.01
    #B, est_A1 = algo1(true_B, A, R, X, Y, Z, cov_X, T, eta, eps, r, lambd, task_function, iterations)

    #save est_A
    #pickle.dump(est_A1, open(estimated_data + "est_A1.pkl", "wb"))

    # ALGO 2
    #print("Executing Algorithm 2...")
=======
    if args.load_data:
        load_data = int(args.load_data)
    else:
        load_data = 1

>>>>>>> 0d91aeda7e0b1427a993232ef0c716b0d0a03839
    #N,_ = X.shape
    #_,d3 = Z.shape
    #Ri = [R[i][task_function[i]] for i in range(N)]
    #est_A2 = algo2(Ri, X, Y, task_function, r, d3, A)
    #print(tl.norm(est_A2-A)/tl.norm(A))
    if load_data:
        A = pickle.load(open(A_and_task_dir + "A.pkl", "rb"))
        X = pickle.load(open(A_and_task_dir + "X.pkl", "rb"))
        Y = pickle.load(open(A_and_task_dir + "Y.pkl", "rb"))
        Z = pickle.load(open(A_and_task_dir + "Z.pkl", "rb"))
        R = pickle.load(open(A_and_task_dir + "R.pkl", "rb"))
        task_function = pickle.load(open(A_and_task_dir + "task_function.pkl", "rb"))
    else:
     # Generate A tensor
        A = generate_A_tensor(d1, d2, d3, r)
        # Generate synthetic training data
        X, Y, Z = generate_training_data(d1, d2, d3, N, T)
        noise = np.random.normal(0, 1, (N, T))
        R = generate_responses(A, X, Y, Z, T)
        #print("R shape = {}".format(R.shape))
        R += noise
        task_function = np.random.randint(0, T, size=N)
        #save generated data
        pickle.dump(A, open(A_and_task_dir + "A.pkl", "wb"))
        pickle.dump(X, open(A_and_task_dir + "X.pkl", "wb"))
        pickle.dump(Y, open(A_and_task_dir + "Y.pkl", "wb"))
        pickle.dump(Z, open(A_and_task_dir + "Z.pkl", "wb"))
        pickle.dump(R, open(A_and_task_dir + "R.pkl", "wb"))
        pickle.dump(task_function, open(A_and_task_dir + "task_function.pkl", "wb"))

    exit()

<<<<<<< HEAD
    #save est_A2
    #pickle.dump(est_A2, open(estimated_data + "est_A2.pkl", "wb"))

    # Save A, est_A, X, Y, Z, task_function, and R
    if (save_data):
        pickle.dump(A, open(save_data + "A.pkl", "wb"))
        pickle.dump(X, open(save_data + "X.pkl", "wb"))
        pickle.dump(Y, open(save_data + "Y.pkl", "wb"))
        pickle.dump(Z, open(save_data + "Z.pkl", "wb"))
        pickle.dump(R, open(save_data + "R.pkl", "wb"))
        pickle.dump(task_function, open(save_data + "task_function.pkl", "wb"))
=======
    true_B = mode_dot(A, Z, mode=2)
    Y_ti = Y[task_function]
    cov_X = np.einsum('bi,bo->bio', X, Y_ti)

    # Perform algorithm 1 to get estimated A
    eps = 0.01
    B, A1, A2 = algo1(true_B, A, R, X, Y, Z, cov_X, T, eta, eps, r, lambd, task_function, iterations)


    #N,_ = X.shape
    #_,d3 = Z.shape
    #Ri = [R[i][task_function[i]] for i in range(N)]
    #A1, A2 = algo2(Ri, X, Y, task_function, r, d3, A)
    #save est_A2
    pickle.dump(A1, open(A_and_task_dir + "est_A1.pkl", "wb"))
    pickle.dump(A2, open(A_and_task_dir + "est_A2.pkl", "wb"))

       # Save A, est_A, X, Y, Z, task_function, and R

>>>>>>> 0d91aeda7e0b1427a993232ef0c716b0d0a03839
