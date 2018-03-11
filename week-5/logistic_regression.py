# LOGISTIC REGRESSION
# In this problem you will create your own target function f (probability in this case) and data set D to see how
# Logistic Regression works. For simplicity, we will take f to be a 0=1 probability so y is a deterministic function
# of x.

# Take d = 2 so you can visualize the problem, and let X = [-1, 1] x [-1, 1] with uniform probability of picking each
# x in X. Choose a line in the plane as the boundary between f(x) = 1 (where y has be to +1) and f(x) = 0 (where y
# has to be -1) by taking two random, uniformly distributed points from X and taking the line passing through them as
# the boundary between y = +1 and y = -1. Pick N = 100 training points at random from X, and evaluate the outputs y_n
# for each of these points x_n.

# Run Logistic Regression with Stochastic Gradient Descent to find g, and estimate E_out/the cross entropy error
# by generating a sufficiently large, separate set of points to evaluate the error. Repeat the experiment for 100
# runs with different targets and take the average. Initialize the weight vector of Logistic Regression to all zeros in
# each run. Stop the algorithm when ||w(t+1) - w(t)|| < 0.01, where w(t) denotes the weight vector at the end of each
# epoch t. An epoch is a full pass through the N data points (use a random permutation of 1, 2, ..., N to present the
# data points to the algorithm within each epoch, and use different permutations for different epochs). Use a learning
# rate of 0.01.

import numpy as np


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


def cross_entropy(X, y, w):
    error = 0
    N = X.shape[0]

    for n in range(N):
        x_n = X[n, :].T
        y_n = y[n]

        s = -y_n*w.T*x_n

        e_n = np.log(1 + np.exp(s))

        error += e_n

    return error/N

eta = 0.01

NUM_ITERATIONS = 100
iterations = np.zeros(NUM_ITERATIONS)

E_out = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    f, g = random_linear_target()
    X, y = generate_data(npoints=100, nvars=2, target=f)

    converged = False
    w_old = np.zeros(3)

    counter = 0

    while not converged:
        counter += 1

        numbers = np.random.permutation(list(range(100)))

        w_temp = w_old

        for n in numbers:
            x_n = np.array(X[n, :]).reshape((3,))
            y_n = np.array(y[n])

            grad = -y_n*x_n/(1 + np.exp(y_n*np.dot(w_old, x_n)))

            w_new = w_temp - eta*grad
            w_temp = w_new

        if np.linalg.norm(w_new - w_old) < 0.01:
            converged = True
            iterations[i] = counter

        w_old = w_new

    w_star = np.matrix(w_new).T

    X_test, y_test = generate_data(npoints=1000, nvars=2, target=f)

    # y_hat = np.sign(X_test*w)

    # NEED TO USE CROSS-ENTROPY ERROR, NOT MISCLASSIFICATION ERROR
    # E_out[i] = np.mean(y_test != y_hat)
    E_out[i] = cross_entropy(X=X_test, y=y_test, w=w_star)

    if (i+1) % 10 == 0:
        print("%i runs completed" % (i+1))

print("Converged in %i iterations on average" % np.mean(iterations))
print("The average out-of-sample error is %.3f" % np.mean(E_out))

# 8. Which of the following is closest to E_out for N = 100?

#   [a] 0.025
#   [b] 0.050
#   [c] 0.075
#   [d] 0.100
#   [e] 0.125

# I get an average out-of-sample (cross-entropy) error 0.101 over 100 runs, which is closest to [d] 0.100
#   CHECK: CORRECT!
print("The average out-of-sample error is %.3f" % np.mean(E_out))


# 9. How many epochs does it take on average for Logistic Regression to converge for N = 100 using the above
# initialization and termination rules and the specified learning rate? Pick the value that is closest to your
# results.

#   [a] 350
#   [b] 550
#   [c] 750
#   [d] 950
#   [e] 1750

# I get an average of 331 epochs over 100 runs, so [a] 350
#   CHECK: CORRECT!
print("Converged in %i iterations on average" % np.mean(iterations))