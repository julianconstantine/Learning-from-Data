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


def g1(x):
    return -1 - 0.05*x[1] + 0.08*x[2] + 0.13*x[1]*x[2] + 1.5*x[1]**2 + 1.5*x[2]**2


def g2(x):
    return -1 - 0.05*x[1] + 0.08*x[2] + 0.13*x[1]*x[2] + 1.5*x[1]**2 + 15*x[2]**2


def g3(x):
    return -1 - 0.05*x[1] + 0.08*x[2] + 0.13*x[1]*x[2] + 15*x[1]**2 + 1.5*x[2]**2


def g4(x):
    return -1 - 1.5*x[1] + 0.08*x[2] + 0.13*x[1]*x[2] + 0.05*x[1]**2 + 0.05*x[2]**2


def g5(x):
    return -1 - 0.05*x[1] + 0.08*x[2] + 1.5*x[1]*x[2] + 0.15*x[1]**2 + 0.15*x[2]**2


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

E_in = np.zeros(1000)

for i in range(1000):
    X, y = generate_data(npoints=1000, nvars=2, target=nonlinear_target())

    # Flip sign for 10% of observations
    index = np.random.choice(a=list(range(1000)), size=100, replace=False)
    index.sort()

    y[index] *= -1

    w = regression_weights(X, y)

    y_hat = np.sign(X*w)

    E_in[i] = np.mean(y_hat != y)

    if i % 100 == 0:
        print(i)

# The average in-sample classification error is 0.508, which is closest to [d] 0.5
#   CHECK: CORRECT!
#   CHECK: This also matches what I got in R
print(np.mean(E_in))


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

errors = np.zeros((1000, 5))

for i in range(1000):

    X, y = generate_data(npoints=1000, nvars=2, target=nonlinear_target())

    # Transform X features
    Z = transform_features(X)

    # Flip sign for 10% of observations
    index = np.random.choice(a=list(range(1000)), size=100, replace=False)
    index.sort()

    # y[index] *= -1

    w_tilde = regression_weights(X=Z, y=y)
    y_tilde = np.sign(Z*w_tilde)

    y1 = np.apply_along_axis(g1, axis=1, arr=X)
    y2 = np.apply_along_axis(g2, axis=1, arr=X)
    y3 = np.apply_along_axis(g3, axis=1, arr=X)
    y4 = np.apply_along_axis(g4, axis=1, arr=X)
    y5 = np.apply_along_axis(g5, axis=1, arr=X)

    errors[i, 0] = np.mean(y1 != y_tilde)
    errors[i, 1] = np.mean(y2 != y_tilde)
    errors[i, 2] = np.mean(y3 != y_tilde)
    errors[i, 3] = np.mean(y4 != y_tilde)
    errors[i, 4] = np.mean(y5 != y_tilde)

    if i % 100 == 0:
        print(i)

print(np.mean(errors, 0))
