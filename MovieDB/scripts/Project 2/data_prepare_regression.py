from data_transform import *
from sklearn import preprocessing

# Extract X-matrix
X = df.values

# Scale X to mean=0 and std=1
X = preprocessing.scale(X)

# Compute values of N, M
N, M = df.shape