#  from exercise 8.2.6

from data_transform import *
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
import scipy.stats as st
import sklearn.linear_model as lm
from toolbox_02450 import rlr_validate


# Extract data
y = df['revenue'].values
y = stats.zscore(np.matrix(y).T)
df_X = df.drop(['revenue', 'vote_average', 'vote_count', 'popularity'],1) * 1#convert bools
X = df_X.values
attributeNames = df_X.columns.tolist()
N, M = df_X.shape

# Normalize data
X = stats.zscore(X)

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True, random_state=1)

MSE_train = np.empty((K,3))
MSE_test = np.empty((K,3))

# Model 1: Values of lambda
lambdas = np.power(10.,range(-10,9))

# Model 2: ANN
n_replicates = 2
n_hidden_units = 2
max_iter = 10000
m_ANN = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss


# Loop through folds
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    
    
    
    
    # Model 1: Linear regression model
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, np.array(y_train.T)[0], lambdas, internal_cross_validation)
    
    m = lm.LinearRegression().fit(X_train, y_train)
    MSE_train[k,1] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    MSE_test[k,1] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
    
    
    # Model 3: Baseline - Compute mean squared error without using the input data at all
    MSE_train[k,2] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    MSE_test[k,2] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
    print('\nOptimal Lambda: {0}'.format(opt_lambda))
    
    
    
    
    # Model 2: ANN
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(m_ANN,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    MSE_test[k,0] = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean


def evaluate(zA, zB):
    # compute confidence interval of model A
    alpha = 0.05
    CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    z = zA - zB
    CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
    
    print('\nCI: {0}'.format(CI))
    print('\nP-value: {0}'.format(p))
    

print('\n\tANN vs Linear Reg')
evaluate(MSE_test[:,0], MSE_test[:,1])

print('\n\tANN vs Baseline')
evaluate(MSE_test[:,0], MSE_test[:,2])

print('\n\tLinear Reg vs Baseline')
evaluate(MSE_test[:,1], MSE_test[:,2])
    
    