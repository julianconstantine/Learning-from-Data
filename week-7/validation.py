################
#  VALIDATION  #
################

# In the following problems, use the data provided in the les in.dta and out.dta for Homework 6. We are going to
# apply linear regression with a nonlinear transformation for classification (without regularization). The nonlinear
# transformation is given by phi_0 through phi_7 which transform (x1, x2) into
#
#           1   x1   x2   x1^2   x2^2   x1*x2   |x1 - x2|   |x1 + x2|
#
# To illustrate how taking out points for validation aects the performance, we will consider the hypotheses trained
# on D_train (without restoring the full D for training after validation is done).

import numpy as np

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
X_out = np.matrix(X_out)

y_in = np.matrix(y_in)
y_out = np.matrix(y_out)


# QUESTION 1: Split in.dta into training (first 25 examples) and validation (last 10 examples). Train on the 25
# examples only, using the validation set of 10 examples to select between five models that apply linear regression
# to phi_0 through phi_k with k = 3, 4, 5, 6, 7. For which model is the classification error on the validation set
# smallest?
#
#   [a] k = 3
#   [b] k = 4
#   [c] k = 5
#   [d] k = 6
#   [e] k = 7


def phi(x):
    z = (x[0], x[1], x[2], x[1]**2, x[2]**2, x[1]*x[2], np.abs(x[1] - x[2]), np.abs(x[1] + x[2]))

    return z

# Split data
X_train, X_val = X_in[0:25], X_in[25:35]
y_train, y_val = y_in[0:25], y_in[25:35]

# Transform training and validation data and cast as matrices
Z_train = np.apply_along_axis(func1d=phi, axis=1, arr=X_train); Z_train = np.matrix(Z_train)
Z_val = np.apply_along_axis(func1d=phi, axis=1, arr=X_val); Z_val = np.matrix(Z_val)

k_values = [3, 4, 5, 6, 7]

weights = {}
E_val = {}

for k in k_values:
    Z_k_train = Z_train[:, 0:(k+1)]
    Z_k_val = Z_val[:, 0:(k+1)]

    # Compute linear regression weights
    w_k = np.linalg.inv(Z_k_train.T*Z_k_train)*Z_k_train.T*y_train

    # Compute predicted values on validation set
    y_k_val = np.sign(Z_k_val*w_k)

    # Compute and store validation error
    E_val[k] = np.mean(y_k_val != y_val)

    # Store weight vector
    weights[k] = w_k

# The smallest validation is 0.0 and occurs at [d] k = 6
#   CHECK: CORRECT!
print(E_val)


# QUESTION 2: Evaluate the out-of-sample classification error using out.dta on the 5 models to see how well the
# validation set predicted the best of the 5 models. For which model is the out-of-sample classification error
# smallest?
#
#   [a] k = 3
#   [b] k = 4
#   [c] k = 5
#   [d] k = 6
#   [e] k = 7

# Out-of-sample classification error
E_out = {}

# Transform out-of-sample data
Z_out = np.apply_along_axis(func1d=phi, axis=1, arr=X_out); Z_out = np.matrix(Z_out)

for k in k_values:
    Z_k_out = Z_out[:, 0:(k+1)]

    w_k = weights[k]

    # Compute predicted values on out-of-sample data
    y_k_hat = np.sign(Z_k_out*w_k)

    # Compute and store out-of-sample classification error
    E_out[k] = np.mean(y_out != y_k_hat)

# The lowest out-of-sample error is 0.084 and occurs at [e] k = 7
#   CHECK: CORRECT!
print(E_out)


# QUESTION 3: Reverse the role of training and validation sets; now training with the last 10 examples and validating
# with the first 25 examples. For which model is the classification error on the validation set smallest?
#
#   [a] k = 3
#   [b] k = 4
#   [c] k = 5
#   [d] k = 6
#   [e] k = 7

# Swap training and validation data
X_train, X_val = X_val, X_train
y_train, y_val = y_val, y_train

# Transform training and validation data and cast as matrices
Z_train = np.apply_along_axis(func1d=phi, axis=1, arr=X_train); Z_train = np.matrix(Z_train)
Z_val = np.apply_along_axis(func1d=phi, axis=1, arr=X_val); Z_val = np.matrix(Z_val)

k_values = [3, 4, 5, 6, 7]

weights = {}
E_val = {}

for k in k_values:
    Z_k_train = Z_train[:, 0:(k+1)]
    Z_k_val = Z_val[:, 0:(k+1)]

    # Compute linear regression weights
    w_k = np.linalg.inv(Z_k_train.T*Z_k_train)*Z_k_train.T*y_train

    # Compute predicted values on validation set
    y_k_val = np.sign(Z_k_val*w_k)

    # Compute and store validation error
    E_val[k] = np.mean(y_k_val != y_val)

    # Store weight vector
    weights[k] = w_k

# Now the smallest validation is still 0.0 and still? occurs at [d] k = 6
#   CHECK: CORRECT!
print(E_val)


# QUESTION 4: Once again, evaluate the out-of-sample classification error using out.dta on the 5 models to see how
# well the validation set predicted the best of the 5 models. For which model is the out-of-sample classification
# error smallest?
#
#   [a] k = 3
#   [b] k = 4
#   [c] k = 5
#   [d] k = 6
#   [e] k = 7

# Out-of-sample classification errors
E_out = {}

for k in k_values:
    Z_k_train = Z_train[:, 0:(k+1)]
    Z_k_out = Z_out[:, 0:(k+1)]

    # Compute linear regression weights
    w_k = weights[k]

    # Compute predicted values on validation set
    y_k_out = np.sign(Z_k_out*w_k)

    # Compute and store validation error
    E_out[k] = np.mean(y_k_out != y_out)


# The lowest out-of-sample error is 0.196, which occurs at [d] k = 6
#   CHECK: CORRECT!
print(E_out)


# QUESTION 5: What values are closest in Euclidean distance to the out-of-sample classification error obtained for
# the model chosen in Problems 1 and 3, respectively?
#
#   [a] 0.0, 0.1
#   [b] 0.1, 0.2
#   [c] 0.1, 0.3
#   [d] 0.2, 0.2
#   [e] 0.2, 0.3

# THe out-of-sample classification for the k=6 model in Problem 1 is 0.072; for the model chosen in Problem 3,
# it is 0.192. This makes sense since we have more training data in Problem 1. Thus, the answer is closest to [b]
# 0.1, 0.2.
#   CHECK: CORRECT!
