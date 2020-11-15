from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
from scipy import stats

from data_prepare_classification import *
from classification_baseline import *
from classification_LR import *
from classification_CT_model import *

X = stats.zscore(X)

K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True,random_state=1)

MSE_test = np.empty((K,3))
error_rates = np.empty((K,3))

def error_measure(y_predicted, y_actual):
    return np.sum(y_predicted != y_actual) / len(y_actual)

for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Baseline model
    m0 = baseline_predict()
    error_rates[k,0] = error_measure(m0 * np.ones(len(y_test)), y_test)
    MSE_test[k,0] = np.square(y_test - m0 * np.ones(len(y_test))).sum() / y_test.shape[0]
    
    # LogReg model
    m1 = model_LR(X_train, y_train)
    error_rates[k,1] = error_measure(m1.predict(X_test), y_test)
    MSE_test[k,1] = np.square(y_test - m1.predict(X_test)).sum() / y_test.shape[0]
    
    # CT model
    m2 = model_CT(X_train, y_train)
    error_rates[k,2] = error_measure(m2.predict(X_test), y_test)
    MSE_test[k,2] = np.square(y_test - m2.predict(X_test)).sum() / y_test.shape[0]

print(np.average(error_rates,0))

def evaluate(zA, zB):
    # compute confidence interval of model A
    alpha = 0.05
    CIA = stats.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=stats.sem(zA))  # Confidence interval

    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    z = zA - zB
    CI = stats.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=stats.sem(z))  # Confidence interval
    p = stats.t.cdf( -np.abs( np.mean(z) )/stats.sem(z), df=len(z)-1)  # p-value
    
    print('\nCI: {0}'.format(CI))
    print('\nP-value: {0}'.format(p))
    
RMSE_test = np.sqrt(np.average(MSE_test,0))

print('\n\tBaseline vs Log Reg')
evaluate(error_rates[:,0], error_rates[:,1])

print('\n\tBaseline vs CT')
evaluate(error_rates[:,0], error_rates[:,2])

print('\n\tLog Reg vs CT')
evaluate(error_rates[:,1], error_rates[:,2])