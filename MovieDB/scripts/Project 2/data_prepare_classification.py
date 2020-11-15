from data_transform import *
from sklearn import preprocessing

# Extract X matrix and y vector
y = df['vote_average'].values.astype(int)

# Extract data
df_X = df.drop(['revenue', 'vote_average', 'vote_count', 'popularity'],1) * 1#convert bools
#X = preprocessing.scale(df_X.values)
X = df_X.values
attributeNames = df_X.columns
N, M = df_X.shape
classNames = list(np.array(range(11)))
attributeNames = list(df_X.columns.values)

# Compute values of C.
C = len(classNames)
