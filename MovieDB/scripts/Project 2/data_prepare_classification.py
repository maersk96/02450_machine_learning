from data_transform import *
from sklearn import preprocessing

# Extract X-matrix
X = df.fillna(0).get_values()

# Scale X to mean=0 and std=1
X = preprocessing.scale(X)

# Classification variabels
y = df['vote_average'].get_values()
# del df['vote_average']

# Compute values of N, M and C.
N = len(y)
#M = len(attributeNames)
#C = len(classNames)
