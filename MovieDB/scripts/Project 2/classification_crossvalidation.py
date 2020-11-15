from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

from data_prepare_classification import *
from classification_baseline import *

K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

MSE_train = np.empty((K,3))
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
    error_rates[k,0] = error_measure(m0 * np.ones(len(y_test), y_test))
    
    # LogReg model
    m1 = model_LR(X_train, y_train)
    error_rates[k,1] = error_measure(m1.predict(X_test), y_test)
    
    # CT model
    m2 = model_CT(X_train, y_train)
    error_rates[k,1] = error_measure(m2.predict(X_test), y_test)


#
print(baseline_predict())
