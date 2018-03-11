# NON-LINEAR TRANSFORMATION
# In these problems, we again apply Linear Regression for classification. Consider the target function:
#
#       f(x_1, x_2) = sgn(x_1**2 + x_2**2 - 0.6)
#
# Generate a training set of N = 1000 points on X = [-1, 1] x [-1, 1] with a uniform probability of picking each x in
# X. Generate simulated noise by flipping the sign of the output in a randomly selected 10% subset of the generated
# training set.

import numpy as np


def nonlinear_target():
    def target(x):
        return np.sign(x[0]**2 + x[1]**2 - 0.6)
    return target


def generate_data(npoints, nvars, target):
    X = np.random.rand(npoints, nvars)*2 - 1
    X = np.matrix(X)

    X = np.append(arr=np.ones((npoints, 1)), values=X, axis=1)

    y = np.apply_along_axis(target, axis=1, arr=X[:, 1:3])
    y = y.reshape((npoints, 1))

    return X, y


def regression_weights(X, y):
    w = np.linalg.inv(X.T*X)*X.T*y

    return w


def transform_features(X):
    Z = np.matrix(np.zeros((X.shape[0], 6)))

    Z[:, 0:3] = X
    Z[:, 3] = np.array(X[:, 1])*np.array(X[:, 2])
    Z[:, 4] = np.array(X[:, 1])**2
    Z[:, 5] = np.array(X[:, 2])**2

    return Z


def make_hypotheses():
    def g1(z):
        return np.sign(-1 - 0.05*z[1] + 0.08*z[2] + 0.13*z[3] + 1.5*z[4] + 1.5*z[5])

    def g2(z):
        return np.sign(-1 - 0.05*z[1] + 0.08*z[2] + 0.13*z[3] + 1.5*z[4] + 15*z[5])

    def g3(z):
        return np.sign(-1 - 0.05*z[1] + 0.08*z[2] + 0.13*z[3] + 15*z[4] + 1.5*z[5])

    def g4(z):
        return np.sign(-1 - 1.5*z[1] + 0.08*z[2] + 0.13*z[3] + 0.05*z[4] + 0.05*z[5])

    def g5(z):
        return np.sign(-1 - 0.05*z[1] + 0.08*z[2] + 1.5*z[3] + 0.15*z[4] + 0.15*z[5])

    return g1, g2, g3, g4, g5


# 8. Carry out Linear Regression without transformation, i.e., with feature vector:
#
#       (1, x1, x2),
#
# to find the weight w. What is the closest value to the classification in-sample error E_in? (Run the experiment 1000
# times and take the average E_in to reduce variation in your results.)
#   [a] 0
#   [b] 0.1
#   [c] 0.3
#   [d] 0.5
#   [e] 0.8

NUM_ITERATIONS = 1000
N = 1000
d = 2

f = nonlinear_target()
E_in = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    X, y = generate_data(npoints=N, nvars=d, target=f)

    # Add noise to y
    index = np.random.choice(a=list(range(N)), size=N//10, replace=False)
    index.sort()

    y[index] *= -1

    w = regression_weights(X, y)

    y_predicted = np.sign(X*w)

    E_in[i] = np.mean(y != y_predicted)

# QUESTION 8 ANSWER
# The average in-sample error is about 0.16-0.17, which is closest to [b] 0.1
#   CHECK: Incorrect, the answer key says [6] 0.5
#   UPDATE: With the correct data-generating function, I now get about 0.5, so we are good!
print("The average in-sample error across 1,000 trials was: " + str(np.mean(E_in)))


# 9. Now, transform the N = 1000 training data into the following nonlinear feature vector:
#
#       (1, x_1, x_2, x_1*x_2, x_1**2, x_2**2)
#
# Find the vector w_tilde that corresponds to the solution of Linear Regression. Which of the following hypotheses is
# closest to the one you find? Closest here means agrees the most with your hypothesis, i.e. has the highest
# probability of agreeing on a randomly selected point. Average over a few runs to make sure your answer is stable.
#   [a] g(x_1, x_2) = sgn(-1 - 0.05*x_1 + 0.08*x_2 + 0.13*x_1*x_2 + 1.5*x_1**2 + 1.5*x_2**2)
#   [b] g(x_1, x_2) = sgn(-1 - 0.05*x_1 + 0.08*x_2 + 0.13*x_1*x_2 + 1.5*x_1**2 + 15*x_2**2)
#   [c] g(x_1, x_2) = sgn(-1 - 0.05*x_1 + 0.08*x_2 + 0.13*x_1*x_2 + 15*x_1**2 + 1.5*x_2**2)
#   [d] g(x_1, x_2) = sgn(-1 - 1.5*x_1 + 0.08*x_2 + 0.13*x_1*x_2 + 0.05*x_1**2 + 0.05*x_2**2)
#   [e] g(x_1, x_2) = sgn(-1 - 0.05*x_1 + 0.08*x_2 + 1.5*x_1*x_2 + 0.15*x_1**2 + 15*x_2**2)

NUM_ITERATIONS = 10000
N, d = 1000, 2

f = nonlinear_target()
h1, h2, h3, h4, h5 = make_hypotheses()

# E_in = np.zeros((NUM_ITERATIONS, 6))

diff = np.zeros((NUM_ITERATIONS, 5))

for i in range(NUM_ITERATIONS):
    X_in, y_in = generate_data(npoints=N, nvars=d, target=f)
    Z_in = transform_features(X_in)

    # Add noise
    index = np.random.choice(a=list(range(N)), size=N//10, replace=False)
    index.sort()

    y_in[index] *= -1

    w_tilde = regression_weights(Z_in, y_in)
    y_tilde_in = np.sign(Z_in*w_tilde)

    y1_in = np.apply_along_axis(h1, axis=1, arr=Z_in)
    y2_in = np.apply_along_axis(h2, axis=1, arr=Z_in)
    y3_in = np.apply_along_axis(h3, axis=1, arr=Z_in)
    y4_in = np.apply_along_axis(h4, axis=1, arr=Z_in)
    y5_in = np.apply_along_axis(h5, axis=1, arr=Z_in)

    # E_in[i, 0] = np.mean(y_tilde_in != y_in)
    diff[i, 0] = np.mean(y1_in != y_tilde_in)
    diff[i, 1] = np.mean(y2_in != y_tilde_in)
    diff[i, 2] = np.mean(y3_in != y_tilde_in)
    diff[i, 3] = np.mean(y4_in != y_tilde_in)
    diff[i, 4] = np.mean(y5_in != y_tilde_in)

    if i % 100 == 0:
        print(str(i) + " iterations completed")

# QUESTION 9 ANSWER
# Hypothesis h3 has the highest agreement with the linear regression weights, so [c]
#   CHECK: Incorrect, the answer key says [a]
#   UPDATE: Averaging over 10,000 iterations, I do (barely) get that h1 is the best hypothesis, so I get option [a]
#           as the answer key says is right. I'm apprehensive about this, however.
print(str(np.mean(a=diff, axis=0)))


# 10. What is the closest value to the classification out-of-sample error E_out of your hypothesis from Problem 9?
# (Estimate it by generating a new set of 1000 points and adding noise, as before. Average over 1000 runs to reduce
# the variation in your results.)
#   [a] 0
#   [b] 0.1
#   [c] 0.3
#   [d] 0.5
#   [e] 0.8

NUM_ITERATIONS = 1000
N, d = 1000, 2

f = nonlinear_target()
h1, h2, h3, h4, h5 = make_hypotheses()

# E_in = np.zeros((NUM_ITERATIONS, 6))

E_out = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    X_out, y_out = generate_data(npoints=N, nvars=d, target=f)
    Z_out = transform_features(X_out)

    # Add noise
    index = np.random.choice(a=list(range(N)), size=N//10, replace=False)
    index.sort()

    y_out[index] *= -1

    y1_out = np.apply_along_axis(h1, axis=1, arr=Z_out)

    E_out[i] = np.mean(y1_out != y_out)

    if i % 100 == 0:
        print(str(i) + " iterations completed")


# QUESTION 10 ANSWER
# I get that the average classification error is about 0.5, which is option [d]
#   CHECK: This is wrong, the key is [b] 0.1
print("The average classification error is " + str(np.mean(E_out)))
