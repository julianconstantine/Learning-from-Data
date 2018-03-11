# LINEAR REGRESSION
# In these problems, we will explore how Linear Regression for classification works. As with the Perceptron Learning
# Algorithm in Homework # 1, you will create your own target function f and data set D. Take d = 2 so you can
# visualize the problem, and assume X = [-1, 1] x [-1, 1] with uniform probability of picking each x in X. In each
# run, choose a random line in the plane as your target function f (do this by taking two randomly, uniformly
# distributed points in [-1, 1] x [-1, 1] and taking the line passing through them), where one side of the line maps
# to +1 and the other maps to -1. Choose the inputs xn of the data set as random points (uniformly in X),
# and evaluate the target function on each x_n to get the corresponding output y_n.

import numpy as np


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

    return target


def generate_data(npoints, nvars, target):
    X = np.random.rand(npoints, nvars)*2 -1
    X = np.matrix(X)

    X = np.append(arr=np.ones((npoints, 1)), values=X, axis=1)

    y = np.apply_along_axis(target, axis=1, arr=X[:, 1:3])
    y = y.reshape((npoints, 1))

    return X, y


def regression_weights(X, y):
    w = np.linalg.inv(X.T*X)*X.T*y

    return w

# 5. Take N = 100. Use Linear Regression to find g and evaluate E_in, the fraction of in-sample points which got
# classified incorrectly. Repeat the experiment 1000 times and take the average (keep the g's as they will be used
# again in Problem 6). Which of the following values is closest to the average E_in? (Closest is the option that
# makes the expression |your answer - given option| closest to 0. Use this definition of closest here and throughout.)
#   [a] 0
#   [b] 0.001
#   [c] 0.01
#   [d] 0.1
#   [e] 0.5

NUM_ITERATIONS = 1000
N = 100
d = 2

E_in = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    f = random_linear_target()
    X, y = generate_data(npoints=N, nvars=d, target=f)

    w = regression_weights(X=X, y=y)
    y_predicted = np.sign(X*w)

    E_in[i] = np.mean(y != y_predicted)

# The average value of E_in was about 0.045, which is closest to [c] 0.01
#   CHECK: CORRECT!
print("The mean in-sample error across 1,000 trials was: " + str(np.mean(E_in)))


# 6. Now, generate 1000 fresh points and use them to estimate the out-of-sample error E_out of g that you got in
# Problem 5 (number of misclassified out-of-sample points / total number of out-of-sample points). Again,
# run the experiment 1000 times and take the average. Which value is closest to the average E_out?
#   [a] 0
#   [b] 0.001
#   [c] 0.01
#   [d] 0.1
#   [e] 0.5

NUM_ITERATIONS = 1000
N_in = 100
N_out = 1000
d = 2

E_out = np.zeros(NUM_ITERATIONS)

for i in range(NUM_ITERATIONS):
    f = random_linear_target()
    X_in, y_in = generate_data(npoints=N_in, nvars=d, target=f)
    X_out, y_out = generate_data(npoints=N_out, nvars=d, target=f)

    w = regression_weights(X=X_in, y=y_in)

    y_out_predicted = np.sign(X_out*w)

    E_out[i] = np.mean(y_out != y_out_predicted)

# Average E_out is about 0.05, which is closest to [c] 0.01 (not 0.1, b/c |0.05-0.1| = 0.05 > |0.05-0.01| = 0.04,
# too bad they don't have 0.05)
#   CHECK: CORRECT!
print("The average out-of-sample error was: " + str(np.mean(E_out)))


# 7. Now, take N = 10. After finding the weights using Linear Regression, use them as a vector of initial weights for
# the Perceptron Learning Algorithm. Run PLA until it converges to a final vector of weights that completely separates
# all the in-sample points. Among the choices below, what is the closest value to the average number of iterations
# (over 1000 runs) that PLA takes to converge? (When implementing PLA, have the algorithm choose a point randomly from
# the set of misclassified points at each iteration)
#   [a] 1
#   [b] 15
#   [c] 300
#   [d] 5000
#   [e] 10000

NUM_ITERATIONS = 1000
N = 10
d = 2

iterations = np.zeros(NUM_ITERATIONS)


for i in range(NUM_ITERATIONS):
    f = random_linear_target()
    X, y = generate_data(npoints=N, nvars=d, target=f)

    w = regression_weights(X=X, y=y)

    correctly_classified = False
    counter = 0

    while not correctly_classified:
        # Run classifier
        y_predicted = np.sign(X*w)

        # Get misclassified points and reshape into 1D NumPy array
        misclassified_index = np.array(y_predicted != y).reshape(N)

        if not misclassified_index.any():
            correctly_classified = True
        else:
            counter += 1

            y_misclassified = y[misclassified_index]
            X_misclassified = X[misclassified_index]

            # Randomly choose index of one misclassified point
            index = np.random.randint(low=0, high=len(y_misclassified), size=1)

            w = w + (y_misclassified[index]*X_misclassified[index]).reshape((d+1, 1))

    iterations[i] = counter

# The average number of iterations required is about 8-9, which is closest to [b] 15
#   CHECK: Incorrect? They say the answer is [a], meaning it's closer to 1 ... I dunno what to do.
#   UPDATE: After fixing my data generating function (it was generating random points on the interval [0, 1] instead
#           of [-1, 1], I now get about 3-4, which is closer to [a] 1 than [b] 15. Yay!
print("Average number of iterations required for convergence: " + str(np.mean(iterations)))
