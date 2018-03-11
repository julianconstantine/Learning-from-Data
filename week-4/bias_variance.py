# BIAS AND VARIANCE
# Consider the case where the target function f: [-1, 1] -> R is given by f(x) = sin(pi*x) and the input probability
# distribution is uniform on [-1 ,1]. Assume that the training set has only two examples (picked independently),
# and that the learning algorithm produces the hypothesis that minimizes the mean squared error on the examples.

import numpy as np

# 4. Assume the learning model consists of all hypotheses of the form h(x) = ax. What is the expected value,
# g_bar(x) of the hypothesis produced by the learning algorithm (expected value with respect to the data set)?
# Express your g_bar(x) = a_hat*x and round a_hat to two decimal digits only, then match exactly to one of the
# following answers.

#   [a] g(x) = 0
#   [b] g(x) = 0.79x
#   [c] g(x) = 1.07x
#   [d] g(x) = 1.58x
#   [e] None of the above

NUM_ITERATIONS = 1000

N = 100000

# Generate N points (x1, y1), (x2, y2)
x1 = np.random.random(N)
x2 = np.random.random(N)

y1 = np.sin(np.pi*x1)
y2 = np.sin(np.pi*x2)

# Find a by minimizing the MSE (with no intercept!) for each pair of points (x1, y1) and (x2, y2)
a = (x1*y1 + x2*y2)/(x1**2 + x2**2)

# Average value of a
a_bar = np.mean(a)

# Average value over 100,000 iterations is 1.43, so choice [e] (None of the above)
#   CHECK: CORRECT!
print("The average value of a is " + str(round(a_bar, 2)))


# 5. What is the closest value to the bias in this case?
#   [a] 0.1
#   [b] 0.3
#   [c] 0.5
#   [d] 0.7
#   [e] 1.0

# The bias is equal to E[(g_bar(x) - f(x))^2]. g_bar(x) = a*x, where a = 0.79 from before; f(x) = sin(pi*x).
# We can plug in a = 0.79 from above into the integral from the expectation and just get bias(g_bar) = 0.204,
# so it is closest to [b] 0.3. But we can approximate the expectation with a sum.

N = 1000
x = 2*np.random.random(N) - 1

# bias(g_bar) = E[(g_bar(x) - f(x))^2]
bias = 1/N*np.sum((a_bar*x - np.sin(np.pi*x))**2)

# The average is about 0.27 (over 1,000 sample points), which is closest to [b] 0.3
#   CHECK: CORRECT!
print("The bias is approximately " + str(bias))


# 6. What is the closest value to the variance in this case?
#   [a] 0.2
#   [b] 0.4
#   [c] 0.6
#   [d] 0.8
#   [e] 1.0

# The variance is given by E_x[E_D[(g(x) - g_bar(x))^2]] = E_x[E_D[(a*xx - a_bar*x)^2]]
# E_D[(a*x - a_bar*x)^2] = E(a^2 - a_bar^2)*x^2 = var(a)*x^2

N_a = 100000

x1 = 2*np.random.random(N_a) - 1
x2 = 2*np.random.random(N_a) - 1

y1 = np.sin(np.pi*x1)
y2 = np.sin(np.pi*x2)

# Find a by minimizing the MSE (with no intercept!) for each pair of points (x1, y1) and (x2, y2)
a = (x1*y1 + x2*y2)/(x1**2 + x2**2)

# Re-calculate a_bar (turns out we don't actually need this!)
a_bar = np.mean(a)

N_x = 100000
x = 2*np.random.random(N_x) - 1

# First, take the expectation w/rt D. From before, we know that
#       E_D[(g(x) - g_bar(x))^2] = var(a)*x^2

# Then, we can calculate E_x[E_D[(g(x) - g_bar(x))^2]] = E_x[var(a)*x^2]
var = np.var(a)*np.mean(x**2)

# The average value of the variance is 0.237, which is closest to [a] 0.2
#   CHECK: CORRECT!
print("The variance is approximately " + str(var))


# 7. Now, let's change H. Which of the following learning models has the least expected value of out-of-sample
# error?

#   [a] Hypotheses of the form h(x) = b
#   [b] Hypotheses of the form h(x) = ax
#   [c] Hypotheses of the form h(x) = ax + b
#   [d] Hypotheses of the form h(x) = ax^2
#   [e] Hypotheses of the form h(x) = ax^2 + b

# Expected OOS error = E_x[E_D[(g_D(x) - f(x))^2]]

# In-sample data
N_in = 10000000

x1 = 2*np.random.random(N_in) - 1
x2 = 2*np.random.random(N_in) - 1

y1 = np.sin(np.pi*x1)
y2 = np.sin(np.pi*x2)

# Out-of-sample data
N_out = 1000000

x = 2*np.random.random(N_out) - 1
y = np.sin(np.pi*x)

# Dictionary containing out-of-sample errors
E_out = {}

# HYPOTHESIS [a]: h(x) = b
# Choose b = 1/2*(y1 + y2)
b = 1/2*(y1 + y2)

bias_a = 0  # (calculated numerically)
var_a = np.mean(b**2)

print(bias_a, var_a)

E_out['a'] = bias_a + var_a


# HYPOTHESIS [b]: h(x) = ax
# Choose a = (x1*y1 + x2*y2)/(x1^2 + x2^2)
a = (x1*y1 + x2*y2)/(x1**2 + x2**2)
a_bar = np.mean(a)

bias_b = np.mean((a_bar*x - y)**2)
var_b = np.var(a)*np.mean(x**2)

print(bias_b, var_b)

E_out['b'] = bias_b + var_b


# HYPOTHESIS [c]: h(x) = ax + b
# Choose a = (y2 - y1)/(x2 - x1)
#        b = (y1*x2 - y2*x1)/(x2 - x1)
x_bar = 1/2*(x1 + x2)
y_bar = 1/2*(y1 + y2)

a = ((x1*y1 + x2*y2) - 2*x_bar*y_bar)/(x1**2 + x2**2 - 2*x_bar**2)
b = y_bar - a*x_bar

a_bar = np.mean(a)
b_bar = np.mean(b)

bias_c = np.mean((a_bar*x + b_bar - y)**2)

# Get covariance of a and b from covariance matrix
cov_ab = np.cov(a, b)[0, 1]

var_c = np.var(b) + 2*cov_ab*np.mean(x) + np.var(a)*np.mean(x**2)

print(bias_c, var_c)

E_out['c'] = bias_c + var_c


# HYPOTHESIS [d]: h(x) = ax^2
# Choose a = (x2*y2 + x1*y1)/(x2**3 + x1**3)
a = ((x2**2)*y2 + (x1**2)*y1)/(x1**4 + x2**4)

a_bar = np.mean(a)

bias_d = np.mean((a_bar*x**2 - y)**2)

var_d = np.var(a)*np.mean(x**4)

print(bias_d, var_d)

E_out['d'] = bias_d + var_d


# HYPOTHESIS [e]: h(x) = ax^2 + b
# We should just be able to replace x1 and x2 with x1^2 and x2^2 in the expressions from [c], but we CAN'T be using
# the simplified formula (where a = delta(y)/delta(x)), we need to use the full formula
x2_bar = 1/2*(x1**2 + x2**2)
y_bar = 1/2*(y1 + y2)

a = ((y1*x1**2 + y2*x2**2) - 2*x2_bar*y_bar)/(x1**4 + x2**4 - 2*x2_bar**2)
b = y_bar - a*x2_bar

# Calculate averages
a_bar = np.mean(a)
b_bar = np.mean(b)

bias_e = np.mean((a_bar*(x**2) + b_bar - y)**2)

# Get covariance of a and b from covariance matrix
cov_ab = np.cov(a, b)[0, 1]

var_e = np.var(b) + 2*cov_ab*np.mean(x**2) + np.var(a)*np.mean(x**4)

print(bias_e, var_e)

E_out['e'] = bias_e + var_e

# As of now, hypothesis [a] h(x) = b has the lowest expected out-of-sample error, but I am getting ridiculous values
# for hypothesis [e] now (it varies from like 100,000-100,000,000), so I don't know if I am doing this right.
#   CHECK: "CORRECT" ([a] is the right answer)
