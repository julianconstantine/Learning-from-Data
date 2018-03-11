# GRADIENT DESCENT
# Consider the nonlinear error surface E(u, v) = (u*exp(v) - 2*v*exp(-u))**2. We start at the point (u, v) =
# (1, 1) and minimize this error using gradient descent in the (u, v) space. Use eta = 0.1 (learning rate,
# not step size).

import numpy as np

from numpy import exp
from numpy.linalg import norm


def gradient(u, v):
    grad_E = np.zeros(2)

    grad_E[0] = 2*(exp(v) + 2*v*exp(-u))*(u*exp(v) - 2*v*exp(-u))
    grad_E[1] = 2*(u*exp(v) - 2*exp(-u))*(u*exp(v) - 2*v*exp(-u))

    return grad_E


def error(u, v):
    return (u*exp(v) - 2*v*exp(-u))**2


# 5. How many iterations (among the given choices) does it take for the error E(u, v) to fall below 10**-14 for the
# first time? In your programs, make sure to use double precision to get the needed accuracy

#   [a] 1
#   [b] 3
#   [c] 5
#   [d] 10
#   [e] 17

# 6. After running enough iterations such that the error has just dropped below 10**-14, what are the closest values
# (in Euclidean distance) among the following choices to the final (u, v) you got in Problem 5?

#   [a] (1.000, 1.000)
#   [b] (0.713, 0.045)
#   [c] (0.016, 0.112)
#   [d] (-0.083, 0.029)
#   [e] (0.045, 0.024)

ETA = 0.1
EPSILON = 10**-14

u0, v0 = 1, 1
grad = gradient(u=u0, v=v0)

counter = 0

while norm(grad)**2 >= EPSILON:
    counter += 1

    u = u0 - ETA*grad[0]
    v = v0 - ETA*grad[1]

    grad = gradient(u=u, v=v)

    u0, v0 = u, v

# QUESTION 5
# It takes 17 iterations to converge, so [e]
#   CHECK: INCORRECT. You need to use the squared error, not the square root of the squared error
#          When I do this, I get 11 iterations, so [d] is correct
print("Took " + str(counter) + " iterations to converge")

# QUESTION 6
# The final value of (u, v) = (0.045, 0.024), so [e]
#   CHECK: CORRECT!
print("The final values of u and v are %.3f and %.3f" % (u, v))


# 7. Now, we will compare the performance of "coordinate descent." In each iteration, we have two steps along the 2
# coordinates. Step 1 is to move only along the u coordinate to reduce the error (assume first-order approximation
# holds like in gradient descent), and step 2 is to reevaluate and move only along the v coordinate to reduce the
# error (again, assume first-order approximation holds). Use the same learning rate of eta = 1.5. What will the error
# E(u, v) be closest to after 15 iterations (30 steps)?

#   [a] 10**-1
#   [b] 10**-7
#   [c] 10**-14
#   [d] 10**-17
#   [e] 10**-21

ETA = 0.1

u0, v0 = 1, 1
grad0 = gradient(u=u0, v=v0)

for i in range(15):
    u = u0 - ETA*grad0[0]

    grad = gradient(u=u, v=v0)

    v = v0 - ETA*grad[1]

    u0, v0 = u, v
    grad0 = gradient(u=u0, v=v0)

# The error is closest to 0.1 so [a]
#   CHECK: CORRECT!
print(error(u, v))
