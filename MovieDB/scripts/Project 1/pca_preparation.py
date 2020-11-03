from data_import import *

import json
import numpy as np


df_pca = df[['runtime', 'budget', 'release_date']]

df_pca['release_date'] = pd.to_datetime(df_pca['release_date'], format="%d/%m/%Y").astype(np.int64) / 10**9


df['genres'] = df['genres'].apply(json.loads)


# K-out-of-one encoding

i = 0
for genres in df['genres']:
    for genre in genres:
        df_pca.loc[i, genre['name']] = 1
    i += 1
    
j = 0
for lang in df['original_language']:
    if lang == 'en':
        df_pca.loc[j, 'english'] = 1
    else:
        df_pca.loc[j, 'english'] = 0
    j += 1   
    
    
# Fill N/A's
df_pca = df_pca.fillna(0)

# Drop bad line
#df_pca = df_pca.drop(4553) # No release data + sad info


# Extract attribute names (1st row, column 4 to 12)
attributeNames = df_pca.columns

# Extract class names to python list,
# then encode with integers (dict)
classLabels = df['vote_average'].astype(str).str.replace('.', '')
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classLabels))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Extract X-matrix
X = df_pca.get_values()

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)