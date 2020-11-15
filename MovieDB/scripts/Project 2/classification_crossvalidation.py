from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np

from data_prepare_classification import *
from classification_baseline import *
from classification_CT_model import *

K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

MSE_train = np.empty((K,3))
MSE_test = np.empty((K,3))

for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Baseline model
    m0 = baseline_predict()
    MSE_train[k,0] = np.square(y_train - m0 * np.ones(len(y_train))).sum() / y_train.shape[0]
    MSE_test[k,0] = np.square(y_test - m0 * np.ones(len(y_test))).sum() / y_test.shape[0]
    
    # LogReg model
    m1 = model_LR(X_train, y_train)
    MSE_train[k,1] = np.square(y_train - m1.predict(X_train)).sum() / y_train.shape[0]
    MSE_test[k,1] = np.square(y_test - m1.predict(X_test)).sum() / y_test.shape[0]
    
    # CT model
    m2 = model_CT(X_train, y_train)
    MSE_train[k,2] = np.square(y_train - m2.predict(X_train)).sum() / y_train.shape[0]
    MSE_test[k,2] = np.square(y_test - m2.predict(X_test)).sum() / y_test.shape[0]

