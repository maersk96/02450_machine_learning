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




#
print(baseline_predict())
