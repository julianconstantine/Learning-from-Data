library(data.table)
library(dplyr)
library(ggplot2)
library(e1071)
# library(kernlab)

makeTarget <- function() {
    f <- function(x) {
        p1 <- runif(n=2, min=-1, max=1)
        p2 <- runif(n=2, min=-1, max=1)
        
        m <- (p2[2] - p1[2])/(p2[1] - p1[1])
        
        y <- p1[2] + m*(x - p1[1])
        
        return(y)
    }
}

makeData <- function(N) {
    f <- makeTarget() 
    
    data <- data.table(x1=runif(n=N, min=-1, max=1), x2=runif(n=N, min=-1, max=1))
    data[, y := ifelse(x2 > f(x1), yes=1, no=-1)]
    
    at_least_one <- abs(sum(data$y)) < N
    
    while (!at_least_one) {
        f <- makeTarget()
        
        data <- data.table(x1=runif(n=N, min=-1, max=1), x2=runif(n=N, min=-1, max=1))
        data[, y := ifelse(x2 > f(x1), yes=1, no=-1)]
        
        at_least_one <- abs(sum(data$y)) < N
    }
    
    return (d)
}

perceptron <- function(data) {
    N <- nrow(data)
    
    X <- cbind(x0=rep(1, times=N), data[, .(x1, x2)]) %>% as.matrix()
    y <- matrix(data$y)
        
    w <- matrix(0, nrow=3, ncol=1)
    
    converged <- FALSE
    
    while (!converged) {
        y_pred <- sign(X %*% w)
        
        misclassified <- which(y != y_pred)
        
        if (any(misclassified)) {
            index <- sample(misclassified, size=1)
            
            w <- w + y[index]*matrix(X[index, ])
        } else {
            converged <- TRUE
        }
    }

    return(w)
}

wfunc <- function(w) {
    f <- function(x) {
        y <- -(w[1] + w[2]*x)/w[3]
        return(y)
    }
    return(f)
}

error_f <- function(g, f) {
    x <- runif(n=1e+6, min=-1, max=1)
    y <- runif(n=1e+6, min=-1, max=1)
    
    y_f <- y > f(x)
    y_g <- y > g(x)
    
    return(mean(y_f != y_g))
}

N <- 10; gucci <- FALSE

while (!gucci) {
    f <- makeTarget()
    
    data <- data.table(x1=runif(n=N, min=-1, max=1), x2=runif(n=N, min=-1, max=1))
    data[, y := ifelse(x2 > f(x1), yes=1, no=-1)]
    
    if (abs(sum(data$y)) < N - 2) {
        gucci <- TRUE
    }
}

w_p <- perceptron(data)
f_p <- wfunc(w_p)

data[, x_p := f_p(x1)]

# If y is a factor, default type is 'C-classification', otherwise must set manually
# NOTE: Setting kernel='linear' fixed all the weird problems I was having with the hyperplane of the SVM not lining up with that of the perceptron from before! Yaaaaay! 
svm.model <- svm(y ~ x1 + x2, data=data, kernel='linear', type='C-classification', cost=2000, scale=FALSE)

w_svm <- t(svm.model$coefs) %*% svm.model$SV
b_svm <- -svm.model$rho

# Concatenate the vectors back to the hyperplane vector
w_svm <- c(b_svm, w_svm)

f_svm <- wfunc(w_svm)

data[, x_svm := f_svm(x1)]

p <- ggplot(data=data) + geom_point(aes(x=x1, y=x2, color=factor(y)))
p <- p + geom_line(aes(x=x1, y=x_p), linetype='dashed')
p <- p + geom_line(aes(x=x1, y=x_svm))
p <- p + ylim(c(-1, 1))
p

N <- 10

errors_p <- numeric(1000)
errors_svm <- numeric(1000)

for (i in 1:1000) {
    gucci <- FALSE
    
    while(!gucci) {
        f <- makeTarget()
        
        data <- data.table(x1=runif(n=N, min=-1, max=1), x2=runif(n=N, min=-1, max=1))
        data[, y := ifelse(x2 > f(x1), yes=1, no=-1)]
        
        if (abs(sum(data$y)) < N) {
            gucci <- TRUE
        }
    }
    
    w_p <- perceptron(data)
    f_p <- wfunc(w_p)
    
    svm.model <- svm(y ~ x1 + x2, data=data, kernel='linear', type='C-classification', cost=1e+6, scale=FALSE)
    
    w_svm <- t(svm.model$coefs) %*% svm.model$SV
    b_svm <- -svm.model$rho
    
    w_svm <- c(b_svm, w_svm)
    
    f_svm <- wfunc(w_svm)
    
    errors_p[i] <- error_f(g=f_p, f=f)
    errors_svm[i] <- error_f(g=f_svm, f=f)
    
    print(i)
}

