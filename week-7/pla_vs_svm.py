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
import pandas as pd
import numpy as np
import cvxopt

def makeTarget():
    x1, y1 = 2*np.random.rand(2) - 1
    x2, y2 = 2*np.random.rand(2) - 1

    a = (y2 - y1)/(x2 - x1)

    b = y1 - a*x1

    def target(x):
        return a*x + b

    return target


def makeData(N):
    data = pd.DataFrame()
    data['x0'] = np.ones(N)
    data['x1'] = np.random.uniform(low=-1, high=1, size=N)
    data['x2'] = np.random.uniform(low=-1, high=1, size=N)

    at_least_one = False

    while not at_least_one:
        f = makeTarget()
        y_temp = data['x2'] > f(data['x1'])
        data['y'] = [1 if y else -1 for y in y_temp]

        if abs(data['y'].sum()) < N:
            at_least_one = True

    return data, f


def makeTestData(M, target):
    test = pd.DataFrame()
    test['x0'] = np.ones(M)
    test['x1'] = np.random.uniform(low=-1, high=1, size=M)
    test['x2'] = np.random.uniform(low=-1, high=1, size=M)

    y_temp = test['x2'] > target(test['x1'])
    test['y'] = [1 if y else -1 for y in y_temp]

    return test


def perceptron(data):
    X = np.matrix(data[['x0', 'x1', 'x2']])
    y = np.matrix(data['y']).T

    w = np.matrix(np.zeros(X.shape[1])).T

    misclassified = y != np.sign(X*w)

    while sum(misclassified) > 0:
        index = np.random.choice(a=np.where(misclassified == True)[0], size=1)[0]

        x_n = X[index, :].T
        y_n = y[index]

        w += x_n*y_n

        misclassified = y != np.sign(X*w)

    # Convert to a nice, easy list
    wlist = w.T.tolist()[0]

    def wfunc(x):
        return -(wlist[0] + wlist[1]*x)/wlist[2]

    return w, wfunc


def solveQP(X, y):
    N = X.shape[0]

    K = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            x_i = X[i, :]
            x_j = X[j, :]

            K[i, j] = float(np.inner(x_i, x_j))

    P = np.outer(y, y)*K
    P = cvxopt.matrix(P)

    q = cvxopt.matrix(-np.ones(N), tc='d')

    G = cvxopt.matrix(-np.identity(N), tc='d')

    h = cvxopt.matrix(np.zeros(N))

    A = cvxopt.matrix(y.T, tc='d')
    b = cvxopt.matrix(0.0, tc='d')

    solution = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)

    return solution


def svm(data):
    X = np.matrix(data[['x0', 'x1', 'x2']])
    y = np.matrix(data['y']).T

    solution = solveQP(X, y)

    alpha = np.ravel(solution['x'])
    alpha = np.matrix(alpha).T

    w = X.T*np.multiply(alpha, y)

    # Recover bias term
    try:
        i = np.where(alpha > 1e-5)[0][0]  # Get index of first point on the margin

        y_i = np.array(y[i])[0][0]
        x_i = np.array(X[i, :])[0]

        # Calculate b from y_i*(w.T*x_i + b) = 1
        bias = 1/y_i - (w[1]*x_i[1] + w[2]*x_i[2])

        w[0] = bias
    except:
        print("error")

    # Convert to a nice, easy list
    wlist = w.T.tolist()[0]

    def wfunc(x):
        return -(wlist[0] + wlist[1]*x)/wlist[2]

    return w, wfunc


# QUESTION 8: For N = 10, repeat the above experiment for 1000 runs. How often is g_SVM better than g_PLA in
# approximating f? The percentage of time is closest to:
#
#   [a] 20%
#   [b] 40%
#   [c] 60%
#   [d] 80%
#   [e] 100%

NUM_ITERATIONS = 1000
N = 10

error_p = np.zeros(NUM_ITERATIONS)
error_s = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    data, target = makeData(N)

    wp, wp_func = perceptron(data)

    ws, ws_func = svm(data)

    x = np.arange(start=-1, stop=1, step=0.1)

    # plt.plot(x, target(x), 'b-', x, wp_func(x), 'k-', x, ws_func(x), 'r-')
    # plt.ylim((-1, 1))
    # plt.close()

    test = makeTestData(M=100000, target=target)

    y_p_temp = test['x2'] > wp_func(test['x1'])
    y_s_temp = test['x2'] > ws_func(test['x1'])

    test['y_p'] = [1 if y else -1 for y in y_p_temp]
    test['y_s'] = [1 if y else -1 for y in y_s_temp]

    error_p[i] = sum(test['y'] != test['y_p'])
    error_s[i] = sum(test['y'] != test['y_s'])

    print(i)

# Out of 1,000 runs, we get that the SVM outperforms the perceptron 606 times for N = 10, which is closest to option
# [c] 60%
#   CHECK: CORRECT!
print("The SVM outperformed the perceptron %i times out of 1,000" % sum(error_s < error_p))


# QUESTION 9: For N = 100, repeat the above experiment for 1000 runs. How often is g_SVM better than g_PLA
# in approximating f? The percentage of time is closest to:
#
#   [a] 10%
#   [b] 30%
#   [c] 50%
#   [d] 70%
#   [e] 90%

NUM_ITERATIONS = 1000
N = 100

error_p = np.zeros(NUM_ITERATIONS)
error_s = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    data, target = makeData(N)

    wp, wp_func = perceptron(data)

    ws, ws_func = svm(data)

    x = np.arange(start=-1, stop=1, step=0.1)

    plt.plot(x, target(x), 'b-', x, wp_func(x), 'k-', x, ws_func(x), 'r-')
    plt.ylim((-1, 1))
    plt.close()

    test = makeTestData(M=100000, target=target)

    y_p_temp = test['x2'] > wp_func(test['x1'])
    y_s_temp = test['x2'] > ws_func(test['x1'])

    test['y_p'] = [1 if y else -1 for y in y_p_temp]
    test['y_s'] = [1 if y else -1 for y in y_s_temp]

    error_p[i] = sum(test['y'] != test['y_p'])
    error_s[i] = sum(test['y'] != test['y_s'])

    print(i)

# Out of 1,000 runs, we get that the SVM outperforms the perceptron 621 times for N = 100, which is closest to option
# [c] 70%
#   CHECK: This isn't right ... honestly just fuck it
print("The SVM outperformed the perceptron %i times out of 1,000" % sum(error_s < error_p))


# QUESTION 10: For the case N = 100, which of the following is the closest to the average number of support vectors
# of g_SVM (averaged over the 1000 runs)?
#
#   [a] 2
#   [b] 3
#   [c] 5
#   [d] 10
#   [e] 20





import numpy as np

secret_number = np.random.randint(low=0, high=100)

guess = -1
num_guesses = 0
has_won = False

while secret_number != guess and num_guesses < 10:
    guess = int(input("Guess a number: "))
    if guess < secret_number:
        print("Your guess is low")
    elif guess > secret_number:
        print("Your guess is high")
    else:
        print("You guessed right!")
        has_won = True

    num_guesses += 1

if has_won:
    print("Congratulations, you won!")
else:
    print("Sorry, you lost")