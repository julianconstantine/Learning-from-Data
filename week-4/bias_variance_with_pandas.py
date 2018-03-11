# BIAS AND VARIANCE
# Consider the case where the target function f: [-1, 1] -> R is given by f(x) = sin(pi*x) and the input probability
# distribution is uniform on [-1 ,1]. Assume that the training set has only two examples (picked independently),
# and that the learning algorithm produces the hypothesis that minimizes the mean squared error on the examples.

import numpy as np
import pandas as pd

import statsmodels.api as sm

# 4. Assume the learning model consists of all hypotheses of the form h(x) = ax. What is the expected value,
# g_bar(x) of the hypothesis produced by the learning algorithm (expected value with respect to the data set)?
# Express your g_bar(x) = a_hat*x and round a_hat to two decimal digits only, then match exactly to one of the
# following answers.

#   [a] g(x) = 0
#   [b] g(x) = 0.79x
#   [c] g(x) = 1.07x
#   [d] g(x) = 1.58x
#   [e] None of the above

N_in = 100000

NUM_ITERATIONS = 1000

# Generate N points (x1, y1), (x2, y2)
x1 = 2*np.random.random(N_in) - 1
x2 = 2*np.random.random(N_in) - 1

y1 = np.sin(np.pi*x1)
y2 = np.sin(np.pi*x2)

# Create empty vector of fitted coefficients
a = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    data = pd.DataFrame()
    data['x'] = [x1[i], x2[i]]
    data['y'] = [y1[i], y2[i]]

    reg = sm.OLS(endog=data['y'], exog=data['x']).fit()

    a[i] = reg.params

    if i % 100 == 0:
        print(i)

# The average value of a is about 1.44, which is [e] none of the above
#   CHECK: CORRECT!
a_bar = np.mean(a)

print(a_bar)


# 5. What is the closest value to the bias in this case?
#   [a] 0.1
#   [b] 0.3
#   [c] 0.5
#   [d] 0.7
#   [e] 1.0

# The bias is equal to E[(g_bar(x) - f(x))^2]. g_bar(x) = a*x, where a = 1.40 from before; f(x) = sin(pi*x).

N_out = 10000

x = 2*np.random.random(N_out) - 1
y = np.sin(np.pi*x)

bias = np.mean((a_bar*x - y)**2)

# The bias is about 0.27, which is closest to [b] 0.3
#   CHECK: CORRECT!
print(bias)


# 6. What is the closest value to the variance in this case?
#   [a] 0.2
#   [b] 0.4
#   [c] 0.6
#   [d] 0.8
#   [e] 1.0

# The variance is given by E_x[E_D[(g(x) - g_bar(x))^2]] = E_x[E_D[(a*xx - a_bar*x)^2]]
# E_D[(a*x - a_bar*x)^2] = E(a^2 - a_bar^2)*x^2 = var(a)*x^2

# Out-of-sample testing points
N = 10000

x = 2*np.random.random(N) - 1
y = np.sin(np.pi*x)

# In-sample training points
x1 = 2*np.random.random(N) - 1
x2 = 2*np.random.random(N) - 1

y1 = np.sin(np.pi*x1)
y2 = np.sin(np.pi*x2)

a = np.zeros(10000)

for i in range(10000):
    data = pd.DataFrame()
    data['x'] = [x1[i], x2[i]]
    data['y'] = [y1[i], y2[i]]

    reg = sm.OLS(endog=data['y'], exog=data['x']).fit()

    a[i] = reg.params

    if i % 100 == 0:
        print(i)

a_bar = np.mean(a)

# Now use this to calculate the variance
var = np.mean((a*x - a_bar*x)**2)

# The variance is about 0.24, which is closest to [a] 0.2
print(var)


# 7. Now, let's change H. Which of the following learning models has the least expected value of out-of-sample
# error?

#   [a] Hypotheses of the form h(x) = b
#   [b] Hypotheses of the form h(x) = ax
#   [c] Hypotheses of the form h(x) = ax + b
#   [d] Hypotheses of the form h(x) = ax^2
#   [e] Hypotheses of the form h(x) = ax^2 + b

NUM_ITERATIONS = 10000

N = 10000

# Out-of-sample testing points
x = 2*np.random.random(N) - 1
y = np.sin(np.pi*x)

# In-sample training points
x1 = 2*np.random.random(N) - 1
x2 = 2*np.random.random(N) - 1

y1 = np.sin(np.pi*x1)
y2 = np.sin(np.pi*x2)

# Empty vectors to store parameters of regression models
reg_a_params = np.zeros(NUM_ITERATIONS)
reg_b_params = np.zeros(NUM_ITERATIONS)
reg_c_params = np.zeros((NUM_ITERATIONS, 2))
reg_d_params = np.zeros(NUM_ITERATIONS)
reg_e_params = np.zeros((NUM_ITERATIONS, 2))

# Run simulation
for i in range(NUM_ITERATIONS):
    data = pd.DataFrame()
    data['1'] = [1, 1]
    data['x'] = [x1[i], x2[i]]
    data['y'] = [y1[i], y2[i]]
    data['xsq'] = data['x']**2

    reg_a = sm.OLS(endog=data['y'], exog=data['1']).fit()
    reg_b = sm.OLS(endog=data['y'], exog=data['x']).fit()
    reg_c = sm.OLS(endog=data['y'], exog=data[['1', 'x']]).fit()
    reg_d = sm.OLS(endog=data['y'], exog=data['xsq']).fit()
    reg_e = sm.OLS(endog=data['y'], exog=data[['1', 'xsq']]).fit()

    reg_a_params[i] = reg_a.params
    reg_b_params[i] = reg_b.params
    reg_c_params[i] = reg_c.params
    reg_d_params[i] = reg_d.params
    reg_e_params[i] = reg_e.params

    if i % 100 == 0:
        print(i)

# Hypothesis [a]: h(x) = b
b = reg_a_params

bias = np.mean((np.mean(b) - y)**2)
var = np.mean((b - np.mean(b))**2)

error_a = {'bias': bias, 'variance': var, 'total': bias+var}

print("HYPOTHESIS [a]: h(x) = b")
print('bias:', bias, 'variance:', var, 'total:', bias+var)

# Hypothesis [b]: h(x) = ax
a = reg_b_params

bias = np.mean((np.mean(a)*x - y)**2)
var = np.mean((a*x - np.mean(a)*x)**2)

error_b = {'bias': bias, 'variance': var, 'total': bias+var}

print("HYPOTHESIS [b]: h(x) = ax")
print('bias:', bias, 'variance:', var, 'total:', bias+var)

# Hypothesis [c]: h(x) = ax + b
b = reg_c_params[:, 0]
a = reg_c_params[:, 1]

bias = np.mean((np.mean(a)*x + np.mean(b) - y)**2)
var = np.mean((np.mean(a)*x + np.mean(b) - a*x - b)**2)

error_c = {'bias': bias, 'variance': var, 'total': bias+var}

print("HYPOTHESIS [c]: h(x) = ax + b")
print('bias:', bias, 'variance:', var, 'total:', bias+var)

# Hypothesis [d]: h(x) = ax^2
a = reg_d_params

bias = np.mean((np.mean(a)*x**2 - y)**2)
var = np.mean((np.mean(a)*x**2 - a*x**2)**2)

error_d = {'bias': bias, 'variance': var, 'total': bias+var}

print("HYPOTHESIS [d]: h(x) = ax^2")
print('bias:', bias, 'variance:', var, 'total:', bias+var)

# Hypothesis [e]: h(x) = ax^2 + b
b = reg_e_params[:, 0]
a = reg_e_params[:, 1]

bias = np.mean((np.mean(a)*x**2 + np.mean(b) - y)**2)
var = np.mean((np.mean(a)*x**2 + np.mean(b) - a*x**2 - b)**2)

error_e = {'bias': bias, 'variance': var, 'total': bias+var}

print("HYPOTHESIS [e]: h(x) = ax^2 + b")
print('bias:', bias, 'variance:', var, 'total:', bias+var)

# QUESTION #7 ANSWER: Hypothesis [b] has the lowest total expected out-of-sample error
#   error [a]: 0.75
#   error [b]: 0.51
#   error [c]: 1.81
#   error [d]: 14.58
#   error [e]: 4351.34
#   CHECK: CORRECT!
