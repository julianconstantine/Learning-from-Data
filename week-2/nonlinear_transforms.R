# NON-LINEAR TRANSFORMS
# In these problems, we again apply Linear Regression for classification. Consider the target function:
#
#       f(x_1, x_2) = sgn(x_1**2 + x_2**2 - 0.6)
#
# Generate a training set of N = 1000 points on X = [-1, 1] x [-1, 1] with a uniform probability of picking each x in
# X. Generate simulated noise by flipping the sign of the output in a randomly selected 10% subset of the generated
# training set.

generate_data <- function(npoints, noise=FALSE) {
    data <- data.frame(X1=runif(n=npoints, min=-1, max=1), X2=runif(n=npoints, min=-1, max=1))
    
    data$y <- sign(data$X1**2 + data$X2**2 - 0.6)
    
    if (noise) {
        index <- sample(1:npoints, size=floor(npoints/10))
        data$y[index] <- -1*data$y[index]
    }
    
    return(data)
}

g1 <- function(data) {
    y_hat <- sign(-1 - 0.05*data$X1 + 0.08*data$X2 + 0.13*data$X1X2 + 1.5*data$X1sq + 1.5*data$X2sq)
    return(y_hat)
}

g2 <- function(data) {
    y_hat <- sign(-1 - 0.05*data$X1 + 0.08*data$X2 + 0.13*data$X1X2 + 1.5*data$X1sq + 15*data$X2sq)
    
    return(y_hat)
}

g3 <- function(data) {
    y_hat <- sign(-1 - 0.05*data$X1 + 0.08*data$X2 + 0.13*data$X1X2 + 15*data$X1sq + 1.5*data$X2sq)
    return(y_hat)
}

g4 <- function(data) {
    y_hat <- sign(-1 - 1.5*data$X1 + 0.08*data$X2 + 0.13*data$X1X2 + 0.05*data$X1sq + 0.05*data$X2sq)
    return(y_hat)
}

g5 <- function(data) {
    y_hat <- sign(-1 - 0.05*data$X1 + 0.08*data$X2 + 1.5*data$X1X2 + 0.15*data$X1sq + 15*data$X2sq)
    return(y_hat)
}


# 8. Carry out Linear Regression without transformation, i.e., with feature vector:
#
#       (1, x1, x2),
#
# to find the weight w. What is the closest value to the classification in-sample error E_in? (Run the experiment 1000 times and take the average E_in to reduce variation in your results.)

#   [a] 0
#   [b] 0.1
#   [c] 0.3
#   [d] 0.5
#   [e] 0.8

errors <- numeric(1000)

for (i in 1:1000) {
    df <- generate_data(npoints=1000, noise=TRUE)
    
    fit <- lm(y ~ X1 + X2, data=df)
    
    y_hat <- sign(predict(fit))
    
    errors[i] <- mean(df$y != y_hat)
    
    if (i %% 100 == 0) {
        print(i)
    }
}

# QUESTION #8 ANSWER: The average (in-sample) misclassification error over 1,000 iterations is about 0.506, which is closest to [d] 0.5
#   CHECK: CORRECT!
mean(errors)


# 9. Now, transform the N = 1000 training data into the following nonlinear feature vector:
#
#       (1, x_1, x_2, x_1*x_2, x_1**2, x_2**2)
#
# Find the vector w_tilde that corresponds to the solution of Linear Regression. Which of the following hypotheses is closest to the one you find? Closest here means agrees the most with your hypothesis, i.e. has the highest probability of agreeing on a randomly selected point. Average over a few runs to make sure your answer is stable.

#   [a] g(x_1, x_2) = sgn(-1 - 0.05*x_1 + 0.08*x_2 + 0.13*x_1*x_2 + 1.5*x_1**2 + 1.5*x_2**2)
#   [b] g(x_1, x_2) = sgn(-1 - 0.05*x_1 + 0.08*x_2 + 0.13*x_1*x_2 + 1.5*x_1**2 + 15*x_2**2)
#   [c] g(x_1, x_2) = sgn(-1 - 0.05*x_1 + 0.08*x_2 + 0.13*x_1*x_2 + 15*x_1**2 + 1.5*x_2**2)
#   [d] g(x_1, x_2) = sgn(-1 - 1.5*x_1 + 0.08*x_2 + 0.13*x_1*x_2 + 0.05*x_1**2 + 0.05*x_2**2)
#   [e] g(x_1, x_2) = sgn(-1 - 0.05*x_1 + 0.08*x_2 + 1.5*x_1*x_2 + 0.15*x_1**2 + 15*x_2**2)

errors <- matrix(0, nrow=1000, ncol=5)

for (i in 1:1000) {
    df <- generate_data(npoints=1000, noise=FALSE)
    
    # Tranaform data
    dfZ <- df
    dfZ$X1X2 <- df$X1*df$X2
    dfZ$X1sq <- dfZ$X1**2
    dfZ$X2sq <- dfZ$X2**2
    
    fit <- lm(y ~ X1 + X2 + X1X2 + X1sq + X2sq, data=dfZ)
    y_hat <- sign(predict(fit))
    
    y_hat1 <- g1(data=dfZ)
    y_hat2 <- g2(data=dfZ)
    y_hat3 <- g3(data=dfZ)
    y_hat4 <- g4(data=dfZ)
    y_hat5 <- g5(data=dfZ)
    
    errors[i, 1] = mean(y_hat1 != y_hat)
    errors[i, 2] = mean(y_hat2 != y_hat)
    errors[i, 3] = mean(y_hat3 != y_hat)
    errors[i, 4] = mean(y_hat4 != y_hat)
    errors[i, 5] = mean(y_hat5 != y_hat)
    
    if (i %% 100 == 0) {
        print(i)
    }
}

# QUESTION #9 ANSWER: Hypothesis [a] (function g1) has the lowest in-sample error (by like a factor of 10!)
#   CHECK: CORRECT!
colMeans(errors)


# 10. What is the closest value to the classification out-of-sample error E_out of your hypothesis from Problem 9? (Estimate it by generating a new set of 1000 points and adding noise, as before. Average over 1000 runs to reduce the variation in your results.)

#   [a] 0
#   [b] 0.1
#   [c] 0.3
#   [d] 0.5
#   [e] 0.8

errors <- numeric(1000)

for (i in 1:1000) {
    df <- generate_data(npoints=1000, noise=TRUE)
    
    dfZ <- df
    dfZ$X1X2 <- df$X1*df$X2
    dfZ$X1sq <- dfZ$X1**2
    dfZ$X2sq <- dfZ$X2**2
    
    y_hat <- g1(data=dfZ)
    
    errors[i] <- mean(dfZ$y != y_hat)
}

# QUESTION #10 ANSWER: The mean out-of-sample classification error (over 1,000 iterations) is 0.143, which is closest to [b] 0.1
#   CHECK: CORRECT!
mean(errors)