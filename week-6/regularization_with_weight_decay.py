######################################
#  REGULARIZATION WITH WEIGHT DECAY  #
######################################

# In the following problems use the data provided in the files in.dta and out.dta as a training and test set
# respectively. Each line of the files corresponds to a two-dimensional input x = (x1, x2) so that X = R^2,
# followed by the corresponding label from Y = {-1, 1}. We are going to apply Linear Regression with a non-linear
# transformation for classication. The nonlinear transformation is given by:
#
#       phi(x1, x2) = (1, x1, x2, x1^2, x2^2, x1x2, |x1 - x2|, |x1 + x2|)
#
# Recall that the classification error is defined as the fraction of misclassified points.

import numpy as np
import pandas as pd

# Read in-sample data
f = open('week-6/in.dta', mode='r')
inlines = f.readlines(); f.close()

X_in = np.ones((35, 3))
y_in = np.ones((35, 1))

for i in range(len(inlines)):
    line = inlines[i]
    line = np.array(line.split(), dtype='float')

    x = line[0:2].reshape(1, 2)
    y = line[2].reshape(1, 1)

    X_in[i, 1:3] = x
    y_in[i] = y

# Read out-of-sample data
f = open('week-6/out.dta', mode='r')
outlines = f.readlines(); f.close()

X_out = np.ones((250, 3))
y_out = np.ones((250, 1))

for i in range(len(outlines)):
    line = outlines[i]
    line = np.array(line.split(), dtype='float')

    x = line[0:2].reshape(1, 2)
    y = line[2].reshape(1, 1)

    X_out[i, 1:3] = x
    y_out[i] = y

# Convert to numpy matrices
X_in = np.matrix(X_in)
y_in = np.matrix(y_in)

X_out = np.matrix(X_out)
y_out = np.matrix(y_out)

# QUESTION 2: Run Linear Regression on the training set after performing the non-linear transformation. What values
# are closest (in Euclidean distance) to the in-sample and out-of-sample classification errors, respectively?
#
#   [a] 0.03, 0.08
#   [b] 0.03, 0.10
#   [c] 0.04, 0.09
#   [d] 0.04, 0.11
#   [e] 0.05, 0.10

def phi(x):
    return x[0], x[1], x[2], x[1]**2, x[2]**2, x[1]*x[2], np.abs(x[1] - x[2]), np.abs(x[1] + x[2])


# Apply non-linear transformation and convert to numpy matrix
Z_in = np.apply_along_axis(func1d=phi, axis=1, arr=X_in)
Z_in = np.matrix(Z_in)

Z_out = np.apply_along_axis(func1d=phi, axis=1, arr=X_out)
Z_out = np.matrix(Z_out)

w_lin = np.linalg.inv(Z_in.T*Z_in)*Z_in.T*y_in

# Calculate in-sample/out-of-sample classification error
y_in_hat = np.sign(Z_in*w_lin)
y_out_hat = np.sign(Z_out*w_lin)

E_in = np.mean(y_in_hat != y_in)
E_out = np.mean(y_out_hat != y_out)

# E_in = 0.029, E_out = 0.084, which is closest to [a] 0.03, 0.08
#   CHECK: CORRECT!
print("The in-sample error is %.3f and the out-of-sample error is %.3f" % (E_in, E_out))


# QUESTION 3: Now add weight decay to Linear Regression, that is, add the term lambda/N*|w|^2 to the squared
# in-sample error, using lambda = 10^k. What are the closest values to the in-sample and out-of-sample classification
# errors, respectively, for k = -3. Recall that the solution for Linear Regression with Weight Decay was derived in
# class.
#
#   [a] 0.01, 0.02
#   [b] 0.02, 0.04
#   [c] 0.02, 0.06
#   [d] 0.03, 0.08
#   [e] 0.03, 0.10

def regularize(X, y, LAMBDA):
    N = X.shape[1]
    w_reg = np.linalg.inv(X.T*X + LAMBDA*np.identity(N))*X.T*y

    return w_reg


def augmented_error(X_in, X_out, y_in, y_out, LAMBDA):
    w_reg = regularize(X=X_in, y=y_in, LAMBDA=LAMBDA)

    # Calculate in-sample/out-of-sample classification error
    y_in_hat = np.sign(X_in*w_reg)
    y_out_hat = np.sign(X_out*w_reg)

    E_in = np.mean(y_in_hat != y_in)
    E_out = np.mean(y_out_hat != y_out)

    return E_in, E_out

E_in_reg, E_out_reg = augmented_error(X_in=Z_in, X_out=Z_out, y_in=y_in, y_out=y_out, LAMBDA=10**-3)

# Using lambda = 10**-3 gives the same errors as before: E_in = 0.029, E_out = 0.080, which is closest to [d] 0.03,
# 0.08
#   CHECK: CORRECT!
print("The in-sample error is %.3f and the out-of-sample error is %.3f" % (E_in_reg, E_out_reg))


# QUESTION 4: Now, use k = 3. What are the closest values to the new in-sample and out-of-sample classication
# errors, respectively?
#
#   [a] 0.2, 0.2
#   [b] 0.2, 0.3
#   [c] 0.3, 0.3
#   [d] 0.3, 0.4
#   [e] 0.4, 0.4

E_in_reg, E_out_reg = augmented_error(X_in=Z_in, X_out=Z_out, y_in=y_in, y_out=y_out, LAMBDA=10**3)

# Now the errors are 0.371 and 0.436 respectively, which is closest to [e] 0.4 and 0.4
#   CHECK: CORRECT!
print("The in-sample error is %.3f and the out-of-sample error is %.3f" % (E_in_reg, E_out_reg))


# QUESTION 5: What value of k, among the following choices, achieves the smallest out-of-sample classification
# error?
#
#   [a] 2
#   [b] 1
#   [c] 0
#   [d] -1
#   [e] -2

k_values = [2, 1, 0, -1, -2]

E_in_reg = {}
E_out_reg = {}

for k in k_values:
    lambda_k = 10**k

    E_in_reg[k], E_out_reg[k] = augmented_error(X_in=Z_in, X_out=Z_out, y_in=y_in, y_out=y_out, LAMBDA=lambda_k)

# The minimum out-of-sample error comes from [d] k = -1
#   CHECK: CORRECT!
print(E_in_reg)
print(E_out_reg)


# QUESTION 6: What value is closest to the minimum out-of-sample classication error achieved by varying k
# (limiting k to integer values)?
#
#   [a] 0.04
#   [b] 0.06
#   [c] 0.08
#   [d] 0.10
#   [e] 0.12

# The minimum out-of-sample error is 0.056, which is closest to [b] 0.06
#   CHECK: CORRECT!
print(E_in_reg)
print(E_out_reg)