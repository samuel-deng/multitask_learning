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
    if args.A_and_task_dir:
        A_and_task_dir = args.A_and_task_dir
    else:
        A_and_task_dir = "meta_test_results/persistent/"
    if args.seed:
        seed = args.seed
    else:
        seed = 42

    A = pickle.load(open(A_and_task_dir + "A.pkl", "rb"))
    X = pickle.load(open(A_and_task_dir + "X.pkl", "rb"))
    Y = pickle.load(open(A_and_task_dir + "Y.pkl", "rb"))
    Z = pickle.load(open(A_and_task_dir + "Z.pkl", "rb"))
    R = pickle.load(open(A_and_task_dir + "R.pkl", "rb"))
    task_function = pickle.load(open(A_and_task_dir + "task_function.pkl", "rb"))
    N,_ = X.shape
    _,d3 = Z.shape
    Ri = [R[i][task_function[i]] for i in range(N)]
    est_A2 = algo2(Ri, X, Y, task_function, r, d3, A)
    #print(tl.norm(est_A2-A)/tl.norm(A))

    #save est_A2
    pickle.dump(est_A2, open(A_and_task_dir + "est_A2.pkl", "wb"))

