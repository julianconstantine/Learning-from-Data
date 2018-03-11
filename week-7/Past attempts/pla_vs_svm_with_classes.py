#################
#  PLA VS. SVM  #
#################

# In the following problems, we compare PLA to SVM with hard margin on linearly separable data sets. For each run,
# you will create your own target function f and data set D. Take d = 2 and choose a random line in the plane as your
# target function f (do this by taking two random, uniformly distributed points on [-1, 1] x [-1, 1] and taking the
# line passing through them), where one side of the line maps to +1 and the other maps to -1. Choose the inputs xn of
# the data set as random points in X = [-1, 1] x [-1, 1], and evaluate the target function on each xn to get the
# corresponding output y_n. If all data points are on one side of the line, discard the run and start a new run.

# Start PLA with the all-zero vector and pick the misclassified point for each PLA iteration at random. Run PLA to nd
# the final hypothesis g_PLA and measure the disagreement between f and g_PLA as P[g_PLA(x) != f(x)] (you can either
# calculate this exactly, or approximate it by generating a sufficiently large, separate set of points to evaluate it).
# Now, run SVM on the same data to nd the nal hypothesis g_SVM by minimizing 1/2*|w|^2 subject to the constraint
# y_n*(w*x_n + b) >= 1 using quadratic programming on the primal or the dual problem. Measure the disagreement
# between f and g_SVM as P[f(x) != g_SVM(x)], and count the number of support vectors you get in each run.

from sklearn import svm

import matplotlib.pyplot as plt
import numpy as np
import cvxopt

def makeTarget():
    x1, y1 = 2*np.random.rand(2) - 1
    x2, y2 = 2*np.random.rand(2) - 1

    a = (y2 - y1)/(x2 - x1)

    b = y1 - a*x1

    def f(x):
        return a*x + b

    return f


def compareMethods(exp):
    p = Perceptron(exp)
    s = SupportVectorMachine(exp)

    p.fit()
    s.fit()

    X_red, X_blue = p.plotData()

    bdry_p = p.fitBoundary()
    bdry_svm = s.fitBoundary()

    plt.plot(X_red[:, 0], X_red[:, 1], 'ro',
             X_blue[:, 0], X_blue[:, 1], 'bo',
             bdry_p[:, 0], bdry_p[:, 1], 'k-',
             bdry_svm[:, 0], bdry_svm[:, 1], 'k.',)


class Experiment:
    def __init__(self, N, d):
        self.f = makeTarget()

        self.X = 2*np.random.rand(N, d) - 1
        self.__X = np.matrix(self.X)
        self.__X = np.append(arr=np.ones((N, 1)), values=self.__X, axis=1)

        y_temp = (self.X[:, 1] > self.f(self.X[:, 0])).tolist()

        self.y = np.array([1 if y == True else -1 for y in y_temp])

        self.__y = np.matrix(self.y).T

    def getMatrixX(self):
        return self.__X

    def getMatrixY(self):
        return self.__y

    def repeat(self, M):
        d = self.X.shape[1]
        X_out = 2*np.random.rand(M, d) - 1
        X_out = np.matrix(X_out)
        X_out = np.append(arr=np.ones((M, 1)), values=X_out, axis=1)

        y_temp = (X_out[:, 2] > self.f(X_out[:, 1])).tolist()

        y_out = np.array([1 if y == True else -1 for y in y_temp])
        y_out = np.matrix(y_out).T

        return X_out, y_out


class Perceptron:
    def __init__(self, exp):
        self.X = exp.X
        self.__X = exp.getMatrixX()

        self.y = exp.y
        self.__y = exp.getMatrixY()

        self.weights = np.zeros(self.__X.shape[1])
        self.__weights = np.matrix(self.weights).T

        self.iterations = 0

    def fit(self):
        correctly_classified = False

        while not correctly_classified:
            # Run classifier
            y_PLA = np.sign(self.__X*self.__weights)

            # Get misclassified points
            misclassified = np.array(y_PLA != self.__y)
            misclassified = misclassified.reshape(self.__X.shape[0])

            if not misclassified.any():
                correctly_classified = True
                self.weights = np.array(self.__weights.T).reshape(self.weights.shape)
            else:
                self.iterations += 1

                y_misclassified = self.__y[misclassified]
                X_misclassified = self.__X[misclassified]

                # Randomly choose index of one misclassified point
                index = np.random.randint(low=0, high=len(y_misclassified), size=1)

                self.__weights += (y_misclassified[index]*X_misclassified[index]).reshape((self.__X.shape[1], 1))

    def plotData(self):
        X_red = self.X[np.where(self.y == -1)]
        X_blue = self.X[np.where(self.y == 1)]

        return X_red, X_blue

    def fitBoundary(self):
        boundary_x = np.linspace(start=-1, stop=1, num=100)
        boundary_y = -(self.weights[0] + self.weights[1]*boundary_x)/self.weights[2]

        boundary = np.zeros((boundary_x.shape[0], 2), dtype='float')
        boundary[:, 0] = boundary_x
        boundary[:, 1] = boundary_y

        return boundary

    def plotFit(self):
        X_red, X_blue = self.plotData()

        bdry = self.fitBoundary()

        plt.plot(X_red[:, 0], X_red[:, 1], 'ro',
                 X_blue[:, 0], X_blue[:, 1], 'bo',
                 bdry[:, 0], bdry[:, 1], 'k-')

    def test(self, X_out, y_out):
        y_pred = np.sign(X_out*self.__weights)

        return np.mean(y_out != y_pred)

    def predict(self):
        return np.sign(self.__X*self.__weights)


class SupportVectorMachine:
    def __init__(self, exp):
        self.X = exp.X
        self.__X = exp.getMatrixX()

        self.y = exp.y
        self.__y = exp.getMatrixY()

        self.weights = np.zeros(self.__X.shape[1])
        self.__weights = np.matrix(self.weights).T

        self.alpha = np.zeros(self.__X.shape[0])
        self.__alpha = np.matrix(self.alpha).T

        self.iterations = 0

    def fit(self):
        # Number of training examples
        N = self.__X.shape[0]

        # Matrix containing kernel dot products
        # K = self.__X*self.__X.T

        K = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                x_i = np.array(self.__X[i])
                x_j = np.array(self.__X[j])

                K[i, j] = float(np.inner(x_i, x_j))

        # Set up matrices for convex optimization solver
        P = cvxopt.matrix(np.outer(self.__y, self.__y)*K)
        q = cvxopt.matrix(-1*np.ones(N))
        A = cvxopt.matrix(self.__y.T, tc='d')
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(-1*np.identity(N))
        h = cvxopt.matrix(np.zeros(N))

        # Compute solution
        svm = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)

        # Lagrange coefficients
        self.alpha = np.ravel(svm['x'])
        self.__alpha = np.matrix(self.alpha).T

        # Weights from SVM solution
        self.__weights = self.__X.T*np.multiply(self.__y, self.__alpha)

        # Store into visible weights vector
        self.weights = np.array(self.__weights.T).reshape(self.weights.shape)

        # Recover bias term
        try:
            i = np.where(self.alpha > 1e-5)[0][0]  # Get index of first point on the margin

            y_i = self.y[i]
            x_i = self.X[i]

            # Calculate b from y_i*(w.T*x_i + b) = 1
            bias = 1/y_i - (self.weights[1]*x_i[0] + self.weights[2]*x_i[1])

            self.weights[0] = bias
            self.__weights[0] = bias
        except IndexError:
            print("All alphas are equal to zero")
        # clf = svm.SVC(C=1e20, kernel='linear')
        # clf.fit(X=self.__X, y=self.__y)

        # self.weights = clf.coef_.reshape(s.weights.shape[0])
        # self.__weights = np.matrix(self.weights).T

    def plotData(self):
        X_red = self.X[np.where(self.y == -1)]
        X_blue = self.X[np.where(self.y == 1)]

        return X_red, X_blue

    def fitBoundary(self):
        boundary_x = np.linspace(start=-1, stop=1, num=100)
        boundary_y = -(self.weights[0] + self.weights[1]*boundary_x)/self.weights[2]

        boundary = np.zeros((boundary_x.shape[0], 2), dtype='float')
        boundary[:, 0] = boundary_x
        boundary[:, 1] = boundary_y

        return boundary

    def plotFit(self):
        X_red, X_blue = self.plotData()

        bdry = self.fitBoundary()

        plt.plot(X_red[:, 0], X_red[:, 1], 'ro',
                 X_blue[:, 0], X_blue[:, 1], 'bo',
                 bdry[:, 0], bdry[:, 1], 'k-')

    def test(self, X_out, y_out):
        y_pred = np.sign(X_out*self.__weights)

        return np.mean(y_out != y_pred)

    def predict(self):
        return np.sign(self.__X*self.__weights)


exp = Experiment(N=10, d=2)
compareMethods(exp)
plt.ylim(-1, 1)
plt.close()


# QUESTION 8: For N = 10, repeat the above experiment for 1000 runs. How often is g_SVM better than g_PLA in
# approximating f? The percentage of time is closest to:
#
#   [a] 20%
#   [b] 40%
#   [c] 60%
#   [d] 80%
#   [e] 100%

N, d = 100, 2
NUM_ITERATIONS = 1000

errors = np.zeros((NUM_ITERATIONS, 2))

for i in range(NUM_ITERATIONS):
    exp = Experiment(N, d)

    p = Perceptron(exp)
    s = SupportVectorMachine(exp)

    p.fit()
    s.fit()

    # misclassified[i, 0] = np.sum(exp.getMatrixY() != p.predict())
    # misclassified[i, 1] = np.sum(exp.getMatrixY() != s.predict())

    X_out, y_out = exp.repeat(M=100000)

    errors[i, 0] = p.test(X_out=X_out, y_out=y_out)
    errors[i, 1] = s.test(X_out=X_out, y_out=y_out)

# The SVM performs better 50.7% of the time, which is closest to [c] 60%
print("The SVM performed better than PLA in %i of 1,000 trials" %
      np.sum(errors[:, 1] < errors[:, 0]))
