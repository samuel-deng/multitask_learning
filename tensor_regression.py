import numpy as np
import tensorly as tl
from tensorly.tenalg import inner
from numpy.linalg import norm
from numpy.linalg import svd

'''
Takes two vectors, x (the user vectors) and y (the observed feature vectors)
and returns a covariate tensor cov_X in R^{d1 x d2 x T}.

Input: x in R^{d1}, y in R^{d2}, t (int), T (number of tasks)
Output: cov_X in R^{d1 x d2 x T}
'''
def generate_covariate_X(x, y, t, T):
    outer = np.outer(x, y)
    cov_X = np.zeros((len(x), len(y), T)) # d1 x d2 x T
    cov_X[:,:, t] = outer
    return cov_X

'''
Takes a tensor B, and computes the Schatten-1 Norm of B 
(average of the nuclear norms of its modes).

Input: B (d1 x d2 x T tensor)
Output: ||B||_S (float)
'''
def schatten1_norm(B):
    total_norm = 0
    for mode in range(B.ndim):
        total_norm += norm(tl.unfold(B, mode), 'nuc')

    total_norm = (1/B.ndim) * total_norm
    return total_norm

'''
Takes a response matrix R (N x T), a list of covariate X (N of them),
and the current iteration's B and outputs the value of the objective function:

    (1/N) sum_{i = 1}^N (R_{i, t(i)} - <X_i, B>)^2 + \lambd ||B||_S

where ||B||_S is the Schatten-1 Norm of B.

Input: R (N x T matrix), cov_X_list (list of N d1xd2xT tensors), B (d1 x d2 x T tensor), lambd (float)
Output: Objective function value (float)
'''
def objective(R, cov_X_list, B, lambd):
    # Main sum
    cost = 0
    for i in range(len(R)): # for i = 1...N
        cost += (R[i][i] - inner(B, cov_X_list[i])) ** 2
    cost = float(1/len(R)) * cost # 1/N sum [(R_i - <X_i, B>)^2]

    # Regularizer
    cost += lambd * schatten1_norm(B)
    return cost

'''
Takes a response matrix R (N x T), a list of covariate X (N of them),
and the current iteration's B and outputs the gradient tensor in d1 x d2 x T:
    
    (2/N) sum_{i = 1}^N (R_{i, t(i)} - <X_i, B>) -X_i + (\lambd/3) * (D_(1) + D_(2) + D_(3))

where that last term is achieved by 3 SVD's, one on each mode of B.

Input: R (N x T matrix), cov_X_list (list of N d1xd2xT tensors), B (d1 x d2 x T tensor), lambd (float)
Output: Gradient Tensor (d1 x d2 x T)
'''
def gradient(R, cov_X_list, B, lambd):
    # Main sum
    gradient = np.zeros(B.shape) # d1 x d2 x T
    for i in range(len(R)):
        gradient += (R[i][i] - inner(cov_X_list[i], B)) * (-1 * cov_X_list[i])
    gradient = (2 * float(1/len(R))) * gradient

    # Regularizer
    for mode in range(B.ndim):
        unfolded_B = tl.unfold(B, mode) # matrix
        print(unfolded_B.shape)
        U, _, V_T = svd(unfolded_B)
        D_mode = U * V_T
        print(D_mode.shape)

    # Take SVD of each unfolding of B (3 of them)
    # Take UV^T of each unfolding 
    # Convert UV^T for each unfolding to a tensor (from d1 x d2T => d1 x d2 x T)

    return gradient

def grad_descent(R, X, Y, T, eta, eps, lambd):
    # Initialize B to a random tensor d1 x d2 x T
    B = np.random.randn(X.shape[1], Y.shape[1], T)

    # Precompute the covariate tensors for each x
    cov_X_list = []
    for i in range(len(X)):
        cov_X_list.append(generate_covariate_X(X[i], Y[i], i, T)) # Y[i] is only if t is the identity function (should really be Y[t(i)])

    print(objective(R, cov_X_list, B, lambd))
    print(gradient(R, cov_X_list, B, lambd).shape)
    
    # Main gradient descent loop
    #while objective(R, cov_X_list, B, lambd) > eps:
        # Calculate gradient
    #    grad = gradient(R, cov_X_list, B, lambd)

        # Update B
    #    B = B - eta * grad

    return B