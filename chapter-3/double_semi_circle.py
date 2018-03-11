# PROBLEM 3.1
# Consider the double semi-circle "toy" learning task below (shown in the book). There are two semi-circles of width
# thk and radius rad, seaprated by sep as shown (red is -1, blue i +1). The center of the top semi-circle is aligned
# with the middle of the edge of the bottom semi-circle. This task is linearly separable when sep >= 0 and not so for
#  sep < 0. Set rad = 100, thk = 5, and sep = 5. Then, generate 2,000 examples uniformly, which means you will have
# approximately 1,000 examples for each class.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

rad, thk, sep = 10, 5, 5


def cross_entropy(w, X, y):
    return np.mean(np.log(1 + np.exp(-np.dot(X, w)*y)))


def semicircular_data(n, rad, thk, sep):
    r1 = rad
    r2 = rad + thk

    theta_red = np.random.uniform(low=0, high=np.pi, size=n)
    r_red = np.random.uniform(low=r1, high=r2, size=n)

    theta_blue = np.random.uniform(low=np.pi, high=2*np.pi, size=n)
    r_blue = np.random.uniform(low=r1, high=r2, size=n)

    y = np.concatenate((np.repeat(a=[-1.0], repeats=n), np.repeat(a=[1.0], repeats=n)))
    X = np.ones(shape=(2*n, 3))

    X[:, 1] = np.concatenate((r_red*np.cos(theta_red), r_blue*np.cos(theta_blue) + rad + thk/2))
    X[:, 2] = np.concatenate((r_red*np.sin(theta_red) + sep/2, r_blue*np.sin(theta_blue) - sep/2))

    return X, y


#   (a) Run the PLA starting from w = 0 until it converges. Plot the data and the final hypothesis.

X, y = semicircular_data(n=1000, rad=rad, thk=thk, sep=sep)

converged = False

weights = np.zeros(3)

t = 0

while not converged:
    h = np.sign(np.dot(X, weights))

    misclassified = y != h

    if not misclassified.any():
        converged = True
    else:
        t += 1

        index = np.where(misclassified)[0]
        random_index = np.random.choice(a=index, size=1)

        x_t = X[random_index, :].reshape(3)
        y_t = y[random_index]

        weights += y_t*x_t

print("Converged in %i iterations" % t)


X_red = X[np.where(y == -1)]
X_blue = X[np.where(y == 1)]

boundary_x = np.linspace(start=-15, stop=30, num=100)
boundary_y = -(weights[0] + weights[1]*boundary_x)/weights[2]

# And you get two lovely semi-circles!
plt.plot(X_red[:, 1], X_red[:, 2], 'ro',
         X_blue[:, 1], X_blue[:, 2], 'bo',
         boundary_x, boundary_y, 'k-')

plt.close()


#   (b) Repeat part (a) using the linear regression (for classification) to obtain w. Explain your observations.

Xmat = np.matrix(X)
ymat = np.matrix(y).T

weights_ols = np.linalg.inv(Xmat.T*Xmat)*Xmat.T*ymat
weights_ols = np.array(weights_ols)

boundary_y_ols = -(weights_ols[0] + weights_ols[1]*boundary_x)/weights_ols[2]

# Plot the boundary from linear regression
plt.plot(X_red[:, 1], X_red[:, 2], 'ro',
         X_blue[:, 1], X_blue[:, 2], 'bo',
         boundary_x, boundary_y_ols, 'k.')

# Plot both boundaries and compare
plt.plot(X_red[:, 1], X_red[:, 2], 'ro',
         X_blue[:, 1], X_blue[:, 2], 'bo',
         boundary_x, boundary_y, 'k-',
         boundary_x, boundary_y_ols, 'k.')


# PROBLEM 3.2
# For the double semi-circle task in Problem 3.1, vary sep in the range {0.2, 0.4, ..., 5}. Generate 2,000 examples
# and run the PLA starting with w = 0. Record the number of iterations PLA takes to converge. Plot sep versus the
# number of iterations taken for PLA to converge. Explain your results.

sep_values = np.arange(start=0.2, stop=5, step=0.2)
iterations_required = []

for sep in sep_values:
    X, y = semicircular_data(n=1000, rad=rad, thk=thk, sep=sep)

    converged = False
    weights = np.zeros(3)

    t = 0

    while not converged:
        h = np.sign(np.dot(X, weights))

        misclassified = y != h

        if not misclassified.any():
            converged = True
        else:
            t += 1

            index = np.where(misclassified)[0]
            random_index = np.random.choice(a=index, size=1)

            x_t = X[random_index, :].reshape(3)
            y_t = y[random_index]

            weights += y_t*x_t

    iterations_required.append(t)
    print(sep, t)

plt.plot(sep_values, iterations_required)


# PROBLEM 3.3
# For the double semi-circle task in Problem 3.1, set seo = -5 and generate 2,000 examples.

#   (a) What will happen if you run PLA on those examples?

#   (b) Run the pocket algorithm for 100,000 iterations and plot E_in versus the iterations number t.


X, y = semicircular_data(n=1000, rad=10, thk=5, sep=-5)

weights = np.zeros(3)

E_in = np.zeros(100000)

for t in range(100000):
    h = np.sign(np.dot(X, weights))

    misclassified = y != h

    index = np.where(misclassified)[0]
    random_index = np.random.choice(a=index, size=1)

    x_t = X[random_index, :].reshape(3)
    y_t = y[random_index]

    weights_temp = weights + y_t*x_t

    error_old = cross_entropy(w=weights, X=X, y=y)
    error_new = cross_entropy(w=weights_temp, X=X, y=y)

    if error_new < error_old:
        weights = weights_temp

    E_in[t] = cross_entropy(w=weights, X=X, y=y)

    if t % 1000 == 0:
        print(t)

plt.plot(np.arange(start=1, stop=100001, step=1), E_in)

plt.close()

#   (c) Plot the data and the final hypothesis in part(b)

