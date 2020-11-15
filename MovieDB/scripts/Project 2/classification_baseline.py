import numpy as np
from data_prepare_classification import *

baseline_classes, baseline_count = np.unique(y, return_counts=True)

def baseline_predict():
    return list(baseline_count).index(max(list(baseline_count)))