from data_transform import *
from sklearn import preprocessing

df = df[df['vote_average'] != 9] # Remove class with 1 member

# Extract X matrix and y vector
y = df['vote_average'].values.astype(int)

# Extract data
df_X = df.drop(['revenue', 'vote_average', 'vote_count', 'popularity'],1) * 1#convert bools
#df_X = df_X[df_X['vote_average'] == 9]
X = preprocessing.scale(df_X.values)
X# = df_X.values
attributeNames = df_X.columns
N, M = df_X.shape
classNames = list(np.array(range(11)))
attributeNames = list(df_X.columns.values)

# Compute values of C.
C = len(classNames)
