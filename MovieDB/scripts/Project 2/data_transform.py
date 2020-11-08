from data_import import *
import json

#Transform Language - Boolean if(eng) = true else false
df['original_language'] = df['original_language'] == 'en'; 
df.rename(columns={'original_language': 'english'}, inplace=True);

#Transform Title to number of characters
df['title'] = df['title'].str.len().fillna(0);

#Transform Tagline to number of characters
df['tagline'] = df['tagline'].str.len().fillna(0);

#Transform Time to Month and Years
df_DateSplit = df['release_date'].str.split('/', expand=True);
df['month'] = df_DateSplit[1];
df['year'] = df_DateSplit[2]; 
del df['release_date'];

#Transform Voting average to int
df['vote_average'] = df['vote_average'].round(0);

#One out of K encoding
def KEncode(column, key):
    return df.drop(column, axis=1).join(pd.get_dummies(
            pd.Series(df[column].apply(json.loads).map(lambda x:[i[key] for i in x])
                .apply(pd.Series).stack().reset_index(1, drop=True)))
                .sum(level=0)
                .add_prefix(column + '_')
                .fillna(0)
           )
    
df = KEncode('genres', 'name') # +20 features
#df = KEncode('production_countries', 'name') # +100 features
#df = KEncode('production_companies', 'name') # +5.000 features
#df = KEncode('keywords', 'name') # +10.000 features

#Remove unnecessary columns
del df['id'] 
del df['spoken_languages'] 
del df['production_companies'] 
del df['production_countries'] 
del df['keywords'] 
del df['homepage'] 
del df['overview']

## Prepare for ML
# Extract X-matrix
X = df.get_values()

# Classification variabels
y = df['vote_average'].get_values()
# del df['vote_average']

# Compute values of N, M and C.
N = len(y)
#M = len(attributeNames)
#C = len(classNames)