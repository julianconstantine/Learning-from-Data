#################
#  PLA VS. SVM  #
#################

# In the following problems, we compare PLA to SVM with hard margin on linearly separable data sets. For each run,
# you will create your own target function f and data set D. Take d = 2 and choose a random line in the plane as your
# target function f (do this by taking two random, uniformly distributed points on [-1, 1] x [-1, 1] and taking the
# line passing through them), where one side of the line maps to +1 and the other maps to -1. Choose the inputs xn of
# the data set as random points in X = [-1, 1] x [-1, 1], and evaluate the target function on each xn to get the
# corresponding output y_n. If all data points are on one side of the line, discard the run and start a new run.

# Start PLA with the all-zero vector and pick the misclassified point for each PLA iteration at random. Run PLA to nd
# the final hypothesis g_PLA and measure the disagreement between f and g_PLA as P[g_PLA(x) != f(x)] (you can either
# calculate this exactly, or approximate it by generating a suciently large, separate set of points to evaluate it).
# Now, run SVM on the same data to nd the nal hypothesis g_SVM by minimizing 1/2*|w|^2 subject to the constraint
# y_n*(w*x_n + b) >= 1 using quadratic programming on the primal or the dual problem. Measure the disagreement
# between f and g_SVM as P[f(x) != g_SVM(x)], and count the number of support vectors you get in each run.

import matplotlib.pyplot as plt
import numpy as np
import cvxopt

def generate_data(npoints, nvars, target):
    X = 2*np.random.rand(npoints, nvars) - 1
    X = np.matrix(X)

    X = np.append(arr=np.ones((npoints, 1)), values=X, axis=1)

    y = np.apply_along_axis(target, axis=1, arr=X[:, 1:3])
    y = y.reshape((npoints, 1))

    return X, y


def random_linear_target():
    x1, y1 = np.random.rand(2)
    x2, y2 = np.random.rand(2)

    a = (y2 - y1)/(x2 - x1)

    b = y1 - a*x1

    def target(x):
        if x[1] > a*x[0] + b:
            return 1
        else:
            return -1

    def divider(x):
        return a*x + b

    return target, divider


def PLAweights(X, y):
    ##################################
    #  FIT WEIGHTS USING PERCEPTRON  #
    ##################################
    w_PLA = np.zeros((d+1, 1))

    correctly_classified = False
    counter = 0

    while not correctly_classified:
        # Run classifier
        y_PLA = np.sign(X*w_PLA)

        # Get misclassified points
        misclassified_PLA = np.array(y_PLA != y)
        misclassified_PLA = misclassified_PLA.reshape(N)

        if not misclassified_PLA.any():
            correctly_classified = True
        else:
            counter += 1

            y_misclassified = y[misclassified_PLA]
            X_misclassified = X[misclassified_PLA]

            # Randomly choose index of one misclassified point
            index = np.random.randint(low=0, high=len(y_misclassified), size=1)

            w_PLA = w_PLA + (y_misclassified[index]*X_misclassified[index]).reshape((d+1, 1))

    return w_PLA


def SVMweights(X, y):
    ###########################
    #  FIT WEIGHTS USING SVM  #
    ###########################

    # QP Problem: Lagrange dual of SVM minimization problem
    #   min(1/2*x.T*P*x + q.T*x)
    #   subject to: A*x = b, G*x <= h

    # Number of training examples
    N = X.shape[0]

    # Matrix containing kernel dot products
    K = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            x_i = np.array(X[i])
            x_j = np.array(X[j])

            K[i, j] = float(np.inner(x_i, x_j))

    # Set up matrices for convex optimization solver
    P = cvxopt.matrix(np.outer(y, y)*K)
    q = cvxopt.matrix(-1*np.ones(N))
    A = cvxopt.matrix(y.T, tc='d')
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(-1*np.identity(N))
    h = cvxopt.matrix(np.zeros(N))

    # Compute solution
    svm = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)

    # Lagrange coefficients
    alpha = np.ravel(svm['x'])

    # Weights from SVM solution
    w_SVM = X.T*np.multiply(y, np.matrix(alpha).T)

    return w_SVM



# QUESTION 8: For N = 10, repeat the above experiment for 1000 runs. How often is g_SVM better than g_PLA in
# approximating f? The percentage of time is closest to:
#
#   [a] 20%
#   [b] 40%
#   [c] 60%
#   [d] 80%
#   [e] 100%

NUM_ITERATIONS = 1000

N = 100
d = 2

iterations = np.zeros(NUM_ITERATIONS)

# Estimated probabilities, first column is for PLA, second is for SVM
probabilities = np.zeros((NUM_ITERATIONS, 2))

for i in range(NUM_ITERATIONS):
    classifier, divider = random_linear_target()
    X, y = generate_data(npoints=N, nvars=d, target=classifier)

    w_PLA = PLAweights(X=X, y=y)
    w_SVM = SVMweights(X=X, y=y)

    X_red = X[np.where(y == 1)[0], :]
    X_blue = X[np.where(y == -1)[0], :]

    boundary_x = np.linspace(start=-1, stop=1, num=100)

    boundary_y_PLA = -(w_PLA[0] + w_PLA[1]*boundary_x)/w_PLA[2]
    boundary_y_PLA = np.array(boundary_y_PLA).reshape(N)

    boundary_y_SVM = -(w_SVM[0] + w_SVM[1]*boundary_x)/w_SVM[2]
    boundary_y_SVM = np.array(boundary_y_SVM).reshape(N)

    plt.plot(X_red[:, 1], X_red[:, 2], 'ro', X_blue[:, 1], X_blue[:, 2], 'bo',
             boundary_x, boundary_y_PLA, 'k-', boundary_x, boundary_y_SVM, 'k.')

    plt.close()

    # Estimate P[f(x) != g(x)] (this is basically calculated out-of-sample error)
    XX, yy = generate_data(npoints=1000, nvars=2, target=classifier)
    yy_PLA = np.sign(XX*w_PLA)
    yy_SVM = np.sign(XX*w_SVM)

    probabilities[i, 0] = np.mean(yy != yy_PLA)
    probabilities[i, 1] = np.mean(yy != yy_SVM)

    iterations[i] = counter

    print(i)




# QUESTION 9: For N = 100, repeat the above experiment for 1000 runs. How often is g_SVM better than g_PLA
# in approximating f? The percentage of time is closest to:
#
#   [a] 10%
#   [b] 30%
#   [c] 50%
#   [d] 70%
#   [e] 90%

# QUESTION 10: For the case N = 100, which of the following is the closest to the average number of support vectors
# of g_SVM (averaged over the 1000 runs)?
#
#   [a] 2
#   [b] 3
#   [c] 5
#   [d] 10
#   [e] 20
