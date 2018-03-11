# NON-LINEAR TRANSFORMATION
# In these problems, we again apply Linear Regression for classification. Consider the target function:
#
#       f(x_1, x_2) = sgn(x_1**2 + x_2**2 - 0.6)
#
# Generate a training set of N = 1000 points on X = [-1, 1] x [-1, 1] with a uniform probability of picking each x in
# X. Generate simulated noise by flipping the sign of the output in a randomly selected 10% subset of the generated
# training set.

import numpy as np
import pandas as pd

# statsmodels.api contains the API for linear regression
import statsmodels.api as sm


def generate_data(npoints, noise=False):
    data = pd.DataFrame()

    data['x0'] = np.ones(npoints)
    data['x1'] = 2*np.random.random(npoints) - 1
    data['x2'] = 2*np.random.random(npoints) - 1

    data['y'] = np.sign(data['x1']**2 + data['x2']**2 - 0.6)

    if noise:
        index = np.random.choice(a=np.arange(0, npoints), size=npoints//10, replace=False)
        data['y'][index] *= -1

    return data

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

E_in = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    data = generate_data(npoints=1000, noise=True)

    reg = sm.OLS(endog=data['y'], exog=data[['x0', 'x1', 'x2']]).fit()

    # Get fitted values by accessing the .fittedvalues attribute
    y_hat = np.sign(reg.fittedvalues)

    # Calculate in-sample error
    E_in[i] = np.mean(data['y'] != y_hat)

    if i % 100 == 0:
        print(i)

# The average in-sample error over 1000 runs is about 0.504, which is closest to [d] 0.5
#   CHECK: CORRECT!
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


def g1(df):
    score = -1 - 0.05*df['x1'] + 0.08*df['x2'] + 0.13*df['x1x2'] + 1.5*df['x1sq'] + 1.5*df['x2sq']
    return np.sign(score)


def g2(df):
    score = -1 - 0.05*df['x1'] + 0.08*df['x2'] + 0.13*df['x1x2'] + 1.5*df['x1sq'] + 15*df['x2sq']
    return np.sign(score)


def g3(df):
    score = -1 - 0.05*df['x1'] + 0.08*df['x2'] + 0.13*df['x1x2'] + 15*df['x1sq'] + 1.5*df['x2sq']
    return np.sign(score)


def g4(df):
    score = -1 - 1.5*df['x1'] + 0.08*df['x2'] + 0.13*df['x1x2'] + 0.05*df['x1sq'] + 0.05*df['x2sq']
    return np.sign(score)


def g5(df):
    score = -1 - 0.05*df['x1'] + 0.08*df['x2'] + 0.13*df['x1x2'] + 0.15*df['x1sq'] + 15*df['x2sq']
    return np.sign(score)

diff = np.zeros((NUM_ITERATIONS, 5))

for i in range(NUM_ITERATIONS):
    df = generate_data(npoints=1000, noise=True)

    # Transform data
    dfZ = df
    dfZ['x1x2'] = dfZ['x1']*dfZ['x2']
    dfZ['x1sq'] = dfZ['x1']**2
    dfZ['x2sq'] = dfZ['x2']**2


    reg = sm.OLS(endog=df['y'], exog=dfZ[['x0', 'x1', 'x2', 'x1x2', 'x1sq', 'x2sq']]).fit()

    # Get fitted values by accessing the .fittedvalues attribute
    y_hat = np.sign(reg.fittedvalues)

    y1_hat = g1(dfZ)
    y2_hat = g2(dfZ)
    y3_hat = g3(dfZ)
    y4_hat = g4(dfZ)
    y5_hat = g5(dfZ)

    # Calculate in-sample error
    diff[i, 0] = np.mean(y_hat != y1_hat)
    diff[i, 1] = np.mean(y_hat != y2_hat)
    diff[i, 2] = np.mean(y_hat != y3_hat)
    diff[i, 3] = np.mean(y_hat != y4_hat)
    diff[i, 4] = np.mean(y_hat != y5_hat)

    if i % 100 == 0:
        print(i)

# Now, hypothesis g1 clearly has the lowest average difference with the predicted values from the linear regression.
# Thus, choice [a] is correct. This corroborates what I got with the R code.
#   CHECK: CORRECT! (obviously)
np.mean(a=diff, axis=0)


# 10. What is the closest value to the classification out-of-sample error E_out of your hypothesis from Problem 9?
# (Estimate it by generating a new set of 1000 points and adding noise, as before. Average over 1000 runs to reduce
# the variation in your results.)

#   [a] 0
#   [b] 0.1
#   [c] 0.3
#   [d] 0.5
#   [e] 0.8

NUM_ITERATIONS = 1000

E_out = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    df = generate_data(npoints=1000, noise=True)

    # Transform data
    dfZ = df
    dfZ['x1x2'] = dfZ['x1']*dfZ['x2']
    dfZ['x1sq'] = dfZ['x1']**2
    dfZ['x2sq'] = dfZ['x2']**2

    y1_hat = g1(dfZ)

    E_out[i] = np.mean(dfZ['y'] != y1_hat)

    if i % 100 == 0:
        print(i)

# The average out-of-sample classification error is 0.143, which is closest to [b] 0.1
#   CHECK: IT'S FINALLY FUCKING CORRECT
print(np.mean(a=E_out, axis=0))
