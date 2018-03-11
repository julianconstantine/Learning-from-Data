# THE PERCEPTRON LEARNING ALGORITHM
# In this problem, you will create your own target function f and data set D to see how the Perceptron Learning
# Algorithm works. Take d = 2 so you can visualize the data, and assume X = [-1, 1] x [-1, 1] with uniform
# probability of picking each x in X.

# In each run, choose a random line in the plane as your target function f (do this by taking two random,
# uniformly distributed points in [-1, 1] x [-1, 1] and taking the line passing through them), where one side of the
# line maps to +1 and the other maps to -1. Choose the inputs xn of the data set as random points (uniformly in X),
# and evaluate the target function on each x_n to get the corresponding output y_n.

# Now, in each run, use the Perceptron Learning Algorithm tofind g. Start the PLA with the weight vector w being all
# zeros (consider sign(0) = 0, so all points are initially misclassified), and at each iteration have the algorithm
# choose a point randomly from the set of misclassified points. We are interested in two quantities: the number of
# iterations that PLA takes to converge to g, and the disagreement between f and g which is P[f(x) != g(x)] (the
# probability that f and g will disagree on their classification of a random point). You can either calculate this
# probability exactly, or approximate it by generating a sufficiently large, separate set of points to estimate it.

# In order to get a reliable estimate for these two quantities, you should repeat the experiment for 1000 runs (each
# run as specified above) and take the average over these runs.

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

    def divider(x):
        return a*x + b

    return target, divider


def generate_data(npoints, nvars, target):
    X = np.random.rand(npoints, nvars)*2 - 1
    X = np.matrix(X)

    X = np.append(arr=np.ones((npoints, 1)), values=X, axis=1)

    y = np.apply_along_axis(target, axis=1, arr=X[:, 1:3])
    y = y.reshape((npoints, 1))

    return X, y


# UGH Y U NO WORK ????
def misclassification_probability(divider, weights):
    # The function divider(x) = a*x + b, need to get a and b
    b = divider(0)
    a = divider(1) - divider(0)

    q = -weights[0]/weights[2]
    p = -weights[1]/weights[2]

    alpha = p - a
    beta = q - b

    p = 1/4*abs(3*(beta**2)/alpha - alpha)

    return p

# 7. Take N = 10. How many iterations does it take on average for the PLA to converge for N = 10 training points?
# Pick the value closest to your results (again, 'closest' means |your answer - given option| is closest to 0).
#   [a] 1
#   [b] 15
#   [c] 300
#   [d] 5000
#   [e] 10000

# 8. Which of the following is closest to P[f(x) != g(x)] for N = 10?
#   [a] 0.001
#   [b] 0.01
#   [c] 0.1
#   [d] 0.5
#   [e] 0.8

NUM_ITERATIONS = 1000

N = 10
d = 2

iterations = np.zeros(NUM_ITERATIONS)

probabilities = np.zeros(NUM_ITERATIONS)  # Estimated probabilities
probabilities2 = np.zeros(NUM_ITERATIONS)  # Calcualted probabilities

for i in range(NUM_ITERATIONS):
    classifier, divider = random_linear_target()
    X, y = generate_data(npoints=N, nvars=d, target=classifier)

    w = np.zeros((d+1, 1))

    correctly_classified = False
    counter = 0

    while not correctly_classified:
        # Run classifier
        y_predicted = np.sign(X*w)

        # Get misclassified points
        misclassified_index = np.array(y_predicted != y)
        misclassified_index = misclassified_index.reshape(N)

        if not misclassified_index.any():
            correctly_classified = True
        else:
            counter += 1

            y_misclassified = y[misclassified_index]
            X_misclassified = X[misclassified_index]

            # Randomly choose index of one misclassified point
            index = np.random.randint(low=0, high=len(y_misclassified), size=1)

            w = w + (y_misclassified[index]*X_misclassified[index]).reshape((d+1, 1))

    # Estimate P[f(x) != g(x)] (this is basically calculated out-of-sample error)
    XX, yy = generate_data(npoints=1000, nvars=2, target=classifier)
    yy_predicted = np.sign(XX*w)

    probabilities[i] = np.mean(yy != yy_predicted)
    probabilities2[i] = misclassification_probability(divider=divider, weights=w)

    iterations[i] = counter


# The average number of iterations required is about 9-10, which is closest to [b] 15
#   CHECK: CORRECT!
print("Average number of iterations required for convergence: " + str(np.mean(iterations)))

# Average of (estimated) P[f(x) != g(x)] is about 0.10-0.11, which is closest to [c] 0.1
#   CHECK: CORRECT!
print("Average probability that f(x) != g(x): " + str(np.mean(probabilities)))

# THIS ONE DOESN'T WORK
# Average of (calcualted) P[f(x) != g(x)] is about 0.11, which is closest to [c] 0.1
# print("Average probability that f(x) != g(x): " + str(np.mean(probabilities2)))


# 9. Now, try N = 100. How many iterations does it take on average for the PLA to converge for N = 100 training
# points? Pick the value closest to your results.
#   [a] 50
#   [b] 100
#   [c] 500
#   [d] 1000
#   [e] 5000

# 10. Which of the following is closest to P[f(x) != g(x)] for N = 100?
#   [a] 0.001
#   [b] 0.01
#   [c] 0.1
#   [d] 0.5
#   [e] 0.8

NUM_ITERATIONS = 1000

N = 100
d = 2

iterations = np.zeros(NUM_ITERATIONS)

probabilities = np.zeros(NUM_ITERATIONS)  # Estimated probabilities
probabilities2 = np.zeros(NUM_ITERATIONS)  # Calculated probabilities

for i in range(NUM_ITERATIONS):
    classifier, divider = random_linear_target()
    X, y = generate_data(npoints=N, nvars=d, target=classifier)

    w = np.zeros((d+1, 1))

    correctly_classified = False
    counter = 0

    while not correctly_classified:
        # Run classifier
        y_predicted = np.sign(X*w)

        # Get misclassified points
        misclassified_index = np.array(y_predicted != y)
        misclassified_index = misclassified_index.reshape(N)

        if not misclassified_index.any():
            correctly_classified = True
        else:
            counter += 1

            y_misclassified = y[misclassified_index]
            X_misclassified = X[misclassified_index]

            # Randomly choose index of one misclassified point
            index = np.random.randint(low=0, high=len(y_misclassified), size=1)

            w = w + (y_misclassified[index]*X_misclassified[index]).reshape((d+1, 1))

    # Estimate P[f(x) != g(x)] (this is basically calculated out-of-sample error)
    XX, yy = generate_data(npoints=1000, nvars=2, target=classifier)
    yy_predicted = np.sign(XX*w)

    probabilities[i] = np.mean(yy != yy_predicted)

    # Calculate P[f(x) != g(x)] exactly

    iterations[i] = counter

# The average number of iterations required is about 95, which is closest to [b] 100
#   CHECK: CORRECT!
print("Average number of iterations required for convergence: " + str(np.mean(iterations)))

# Average of P[f(x) != g(x)] is about 0.01, which is closest to [b] 0.01
#   CHECK: CORRECT!
print("Average probability that f(x) != g(x): " + str(np.mean(probabilities)))
