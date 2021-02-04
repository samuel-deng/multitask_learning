import argparse
import numpy as np
import pickle
from generate_data import generate_training_data
from generate_data import generate_responses
from generate_data import generate_A_tensor

if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", help="First dimension (d1).")
    parser.add_argument("--d2", help="Second dimension (d2).")
    parser.add_argument("--d3", help="Third dimension (d3).")
    parser.add_argument("--T", help="Number of tasks (T).")
    parser.add_argument("--r", help="CP Rank of the underlying tensor A.")
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
    if args.T:
        T = int(args.T)
    else:
        T = 1
    if args.r:
        r = int(args.r)
    else:
        r = 10
    if args.A_and_task_dir:
        A_and_task_dir = args.A_and_task_dir
    else:
        A_and_task_dir = "meta_test_results/persistent/"
    if args.seed:
        seed = args.seed
    else:
        seed = 42

    # Generate A tensor
    A = generate_A_tensor(d1, d2, d3, r, seed)

    # Generate Y0 and Z0 task 
    Y_mu = 0
    Y_sigma = 1/np.sqrt(d2)
    Z_mu = 0
    Z_sigma = 1/np.sqrt(d3)
    Y0 = Y_sigma * np.random.randn(d2) + Y_mu
    Z0 = Z_sigma * np.random.randn(d3) + Z_mu

    # Save
    pickle.dump(A, open(A_and_task_dir + "A.pkl", "wb"))
    pickle.dump(Y0, open(A_and_task_dir + "Y0.pkl", "wb"))
    pickle.dump(Z0, open(A_and_task_dir + "Z0.pkl", "wb"))
