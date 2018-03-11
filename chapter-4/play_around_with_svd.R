library(data.table)
library(dplyr)

################################
#     INITIAL EXPLORATIONS     #
################################

data <- data.table(x1=runif(n=100, min=0, max=5), x2=runif(n=100, min=-2, max=2))
data[, x3 := x2-1]
data[, y := 2 + 3*x1 + 2*x2 + x3 + rnorm(n=100, mean=0, sd=2)]

# Plot the data
with(data, {plot(x1, y)})
with(data, {plot(x2, y)})

# Perform principal components analysis
# NOTE: The formula (first argument) cannot have a response variable
pca <- prcomp( ~ x1 + x2 + x3, data=data, center=TRUE, scale.=TRUE)

# Stadard deviations of principal components (singular values)
pca$sdev

# principal components
pca$rotation

# NOTE: If we set tol=1e-10, then anything with a standard deviation <= 1e-5 will be dropped
# Thus, we will only see 2 principal components (as expected) because x3 is actually just a multiple of x1
pca2 <- prcomp( ~ x1 + x2 + x3, data=data, center=TRUE, scale.=TRUE, tol=1e-5)

# The standard deviations and PC vectors 1 and 2 have not changed
pca2


######################################
#     LIGHT SOURCES PCA TUTORIAL     #
######################################

# Locations of recorders A, B, C, and D (in four corners of unit box)
recorders <- data.table(X=c(0, 0, 1, 1), Y=c(0, 1, 1, 0), row.names=LETTERS[1:4])

# Locations of centers of light sources inside the box
centers <- data.table(X=c(0.3, 0.5), Y=c(0.8, 0.2))

# Intensities of sinusoidal and cosinusoidal light
intensities <- data.table(sine=sin((0:99)*pi/10) + 1.2, cosine=0.7*cos((0:99)*pi/10) + 0.9)

# Create 2x4 empty matrix to contain distances
dists <- matrix(nrow=dim(centers)[1], ncol=dim(recorders)[1], dimnames=list(NULL, row.names(recorders)))

# Loop through each of the four corner points and calculate the distances to the light sources
for (i in 1:dim(dists)[2]) {
    dists[, i] <- sqrt((centers$X - recorders$X[i])**2 + (centers$Y - recorders$Y[i])**2)
}

# Set the seed
set.seed(500)

# Compute the recorded light intensity data. NOTE: You can use the jitter() function to add noise to the data 
recorded_data <- data.table(jitter(as.matrix(intensities) %*% as.matrix(exp(-2*dists)), amount=0))

# Plot the variables against each other
plot(recorded_data)

# The data is very highly correlated (esp. A and D, B and C)
cor(recorded_data) %>% round(digits=2)

# Time series plot (wow, I'm surprised it figured everything out automatically)
# NOTE: plot.ts() just assumes that the rows indicate time intervals and the columns are observations
plot.ts(recorded_data)

#########################################
# principal COMPONENTS ANALYSIS BY HAND #
#########################################
# Idea: Find the eigenvectors of the covariance matrix X'X and use the spectral decomposition

# Mean-normalize the data (PCA assumes that the data is mean normalized, but not scaled by standard dev.)
X <- as.matrix(recorded_data)
X <- X - matrix(rep(colMeans(X), each=dim(X)[1]), nrow=dim(X)[1])

# Calculate covariance matrix
A <- t(X) %*% X

# Diagonalize A = E*D*t(E)
E <- eigen(A, symmetric=TRUE)

# Obtain principal components vectors (rows of matrix E$vectors)
# The columns of P are the principal components
P <- t(E$vectors)

# Get the "new" data and the standard deviations from the principal components
newdata <- X %*% P

####################################################
# BUILT-IN principal COMPONENTS ANALYSIS FUNCTIONS #
###3################################################
# There are two function: prcomp() and princomp():
#   prcomp() performs PCA by using the singular value decomposition (SVD)
#   princimp() performs PCA by using the spectral decomposition
# The SVD is supposed to be more accurate, so prcomp() is preferred

# prcomp() can be run on your data frame and returns a "prcomp" class object
pr <- prcomp(recorded_data)

# If you plot a prcomp object, it will automatically create a plot of the variances of the principal components
# We see that two of the components have negligible variances, which is what we expected since there are only two light sources and four original measurements
plot(pr)

# Plot relative to size of first principal component
barplot(pr$sdev/pr$sdev[1])

# If we set tol=0.1, then any principal compinent whose standard deviation (singular value) is less than 10% of the standard deviation (singular value) of the first principal component will be dropped
pr2 <- prcomp(recorded_data, tol=0.1)

# Plot the two principal components as a time series
plot.ts(pr2$x)
