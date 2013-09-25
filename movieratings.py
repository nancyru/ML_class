# movieratings.py
# This is from/inspired by "Introduction to Machine Learning" by Andrew Ng on Coursera.
# The class was conducted in Matlab.  I converted the recommender systems exercise into
# python.  This was part 2 of homework exercise 8.

# Works best with ipython pylab

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

def loaddata(filename):
    d = np.loadtxt(filename, delimiter = ',')
    return d

def loadMovieList():
    filename = 'movie_ids.txt'
    with open(filename) as f:
        movieList = f.readlines()
    return movieList         

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lamb):
    ### Compute cost function using vectorization
    
    #unpack X, Theta, from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)
    
    #calculate cost function (vectorized)
    J = 1./2*np.sum(np.power(np.dot(X, Theta.T)-Y, 2)*R) + lamb/2. * (np.sum(np.power(Theta,2)) + np.sum(np.power(X,2)))
    return J 
    
def cofiCostFuncUnvectorized(params, Y, R, num_users, num_movies, num_features, lamb):
    ### Compute cost function without vectorization
      
    #unpack X, Theta, from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)
    
    #compute cost function
    J_noreg = 0
    for i in range(0, num_movies):
        for j in range(0, num_users):
            if R[i,j] == 1:
                J_noreg += 0.5* np.power(np.dot(Theta[j],X[i])-Y[i,j], 2)
    
    J = J_noreg + lamb/2.* (np.sum(np.power(Theta,2)) + np.sum(np.power(X,2)))   
    return J
 
def cofiCostFuncSparse(params, Y, num_users, num_movies, num_features, lamb):
    ### Calculate cost function using sparse matrices
    
    #unpack X, Theta, from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)
    
    import itertools
    import scipy
    Y = scipy.sparse.coo_matrix(Y)
    
    J_noreg = 0
    for i, j, v in itertools.izip(Y.row, Y.col, Y.data):
        J_noreg += 0.5* np.power(np.dot(Theta[j],X[i])-v, 2)   
    J = J_noreg + lamb/2.* (np.sum(np.power(Theta,2)) + np.sum(np.power(X,2)))   
    return J
     
def cofiCostFuncGrad(params, Y, R, num_users, num_movies, num_features, lamb):
    ###Calculate gradient of cost function using vectorization
    
    #unpack X, Theta, from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)  
    
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    for k in range(0,num_features):
        X_grad[:,k] =  np.dot(((np.dot(X, Theta.T)-Y)*R), Theta[:,k]) + lamb*X[:,k]
        Theta_grad[:,k] = np.dot(((np.dot(X, Theta.T)-Y)*R).T,X[:,k]) + lamb*Theta[:,k]
    grad = np.concatenate((X_grad.flatten(), Theta_grad.flatten()))  
    return grad 
    
def cofiCostFuncGradSparse(params, Y, num_users, num_movies, num_features, lamb):
    ###Calculate gradient of cost function using sparse matrices
    import itertools
    import scipy
    X = params[0:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    
    Y = scipy.sparse.coo_matrix(Y)
    for k in range(0,num_features):
        X_sum = 0
        Theta_sum = 0
        for i, j, v in itertools.izip(Y.row, Y.col, Y.data):
            coeff = np.dot(Theta[j], X[i])-v
            X_grad[i,k] += coeff*Theta[j, k]
            Theta_grad[j, k] += coeff * X[i, k] 
        #Add regularization
        X_grad[:, k] += lamb*X[:,k]
        Theta_grad[:,k] += lamb*Theta[:,k]
    grad = np.concatenate((X_grad.flatten(), Theta_grad.flatten()))  
    return grad   

def computeNumericalGradient(J, params):
# This function was provided by the machine learning class for our use in
# the homework assignments.  Here I have converted it from Matlab.

# %COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
# %and gives us a numerical estimate of the gradient.
# %   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
# %   gradient of the function J around theta. Calling y = J(theta) should
# %   return the function value at theta.

# % Notes: The following code implements numerical gradient checking, and 
# %        returns the numerical gradient.It sets numgrad(i) to (a numerical 
# %        approximation of) the partial derivative of J with respect to the 
# %        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
# %        be the (approximately) the partial derivative of J with respect 
# %        to theta(i).)
# %                
    perturb = np.zeros(params.size)
    numgrad = np.zeros(params.size)
    e = 1e-1    
    for p in range(0, np.size(params)):
        perturb[p] = e
        loss1 = J(params - perturb)
        loss2 = J(params + perturb)
        numgrad[p] = (loss2 - loss1)/(2.*e)
        perturb[p] = 0 
    return numgrad     
    
def checkCostFunction(lamb = 0):
    #Crate small problem
    X_t = np.random.random((4, 3))
    Theta_t = np.random.random((5, 3))

    #Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.random((Y.shape)) > 0.5] = 0
    R = np.zeros((Y.shape))
    R[Y != 0] = 1
    
    # Run Gradient Checking
    X = np.random.normal(0, 1, (X_t.shape))
    Theta = np.random.normal(0, 1, (Theta_t.shape))
    num_users = np.size(Y, 1)
    num_movies = np.size(Y, 0)
    num_features = np.size(Theta_t, 1)
    
    params = np.concatenate((X.flatten(), Theta.flatten()))
    myfunc = lambda params: cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lamb)
    numgrad = computeNumericalGradient(myfunc, params)
    grad = cofiCostFuncGrad(params, Y, R, num_users, num_movies, num_features, lamb)    
    for i in range(0, grad.size):
        print (numgrad[i], grad[i] )
    
 
def normalizeRatings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)
    for i in range(0, m):
        idx = np.nonzero(R[i,:])
        Ymean[i] = np.mean(Y[i,idx])
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]
    return (Ynorm, Ymean)    
    
## =============== Loading movie ratings dataset ================

Y = loaddata('movies_Y.csv')
R = loaddata('movies_R.csv')    

# Plot 'image' of Y
imgplot = plt.imshow(Y)
plt.show()

## ==== Collaborative Filtering Cost Function and Gradient===========

# Compute cost function and test with pretrained weights
X = loaddata('movieParams_X')
Theta = loaddata('movieParams_Theta')
num_users = loaddata('movieParams_num_users')
num_movies = loaddata('movieParams_num_movies')
num_features = loaddata('movieParams_num_features')

# Reduce the data set size so that this runs faster
num_users, num_movies, num_features = 4, 5, 3 
X = X[0:num_movies, 0:num_features];
Theta = Theta[0:num_users, 0:num_features];
Y = Y[0:num_movies, 0:num_users];
R = R[0:num_movies, 0:num_users];

#  Evaluate cost function
lamb = 1.5
params = np.concatenate((X.flatten(), Theta.flatten()))
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lamb)
grad = cofiCostFuncGrad(params, Y, R, num_users, num_movies, num_features, lamb)
print 'Cost Function: ', J
print 'Gradient: ', grad 

# Compare cost function with numerically calculated cost function
print 'Checking cost function...'
checkCostFunction(lamb)

## ==== Try Other Versions of Collaborative Filtering Cost Function and Gradient========

# Unvectorized form of cost function
J_unvec = cofiCostFuncUnvectorized(params, Y, R, num_users, num_movies, num_features, lamb)
print 'Cost Function (unvectorized) ', J_unvec

# Cost function and Gradient using sparse matrices
J_sparse = cofiCostFuncSparse(params, Y, num_users, num_movies, num_features, lamb)
print 'Cost Function (sparse matrices)', J_sparse
grad_sparse =  cofiCostFuncGradSparse(params, Y, num_users, num_movies, num_features, lamb)
print 'Gradient (sparse matrices)', grad_sparse

## ============== Entering ratings for a new user ===============
# Train on movie list
movieList = loadMovieList()

# Add ratings for new user
my_ratings = np.zeros(1682)
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4;
my_ratings[63]= 5;
my_ratings[65]= 3;
my_ratings[68] = 5;
my_ratings[182] = 4;
my_ratings[225] = 5;
my_ratings[354]= 5;

## ================== Learning Movie Ratings ====================
Y = loaddata('movies_Y.csv')
R = loaddata('movies_R.csv')   

#  Add our own ratings to the data matrix
Y = np.concatenate((my_ratings.reshape(np.size(my_ratings),1), Y), axis = 1)
R = np.concatenate(((my_ratings!=0).reshape(np.size(my_ratings),1),R), axis = 1)

#  Normalize ratings
(Ynorm, Ymean) = normalizeRatings(Y, R)

#  Useful values
num_users = np.size(Y, 1);
num_movies = np.size(Y, 0);
num_features = 10;

#  Set Initial Parameters (Theta, X)
X = np.random.normal(0, 1, (num_movies, num_features))
Theta = np.random.normal(0, 1, (num_users, num_features))
initial_params = np.concatenate((X.flatten(), Theta.flatten()))

#  Set regularization
lamb = 10

#  Set arguments and callback function
myargs = (Ynorm, R, num_users, num_movies, num_features, lamb)
def my_callback(x):
    #  Print value of cost function
    print cofiCostFunc(x, Ynorm, R, num_users, num_movies, num_features, lamb)
    return
    
#  Optimize    
print 'begin optimization...'    
params = optimize.fmin_cg(f = cofiCostFunc, x0 = initial_params, fprime = cofiCostFuncGrad, args = myargs, maxiter = 300, full_output = True, disp = True, callback = my_callback)

#  Unfold the returned theta, X
X = params[0][0:num_movies*num_features].reshape(num_movies, num_features)
Theta = params[0][num_movies*num_features:].reshape(num_users, num_features)

## ================== Recommendation for you ====================
p = np.dot(X, Theta.T);
my_predictions = p[:,0] + Ymean;

r = np.argsort(my_predictions)

movieList = loadMovieList();

print 'Top recommendations for you:'
for i in range(1, 21):
    print 'Predicting rating ', '%.2f' % my_predictions[r[-i]], 'for movie ', movieList[r[-i]]

print 'Original ratings provided:'
for i in range(0, my_ratings.size):
    if my_ratings[i] > 0:
        print 'Rated ', '%d' % my_ratings[i], 'for ', movieList[i]
