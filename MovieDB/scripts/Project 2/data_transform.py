from data_import import *
from KEncode import *
import numpy as np

#Transform Language - Boolean if(eng) = true else false
df['original_language'] = df['original_language'] == 'en'; 
df.rename(columns={'original_language': 'english'}, inplace=True);

#Transform Title to number of characters
df['title'] = df['title'].str.len().fillna(0);

#Transform Tagline to number of characters
df['tagline'] = df['tagline'].str.len().fillna(0);

#Transform Time to Month and Years
df.dropna(axis=0,subset=['release_date'], inplace=True);
df_DateSplit = df['release_date'].str.split('/', expand=True);
df['month'] = df_DateSplit[1].astype('int32');
df['year'] = df_DateSplit[2].astype('int32'); 
del df['release_date'];

#Transform Voting average to int (our class)
df['vote_average'] = df['vote_average'].round(0);


#KEncode
df = KEncode(df, 'genres', 'name', 50) # +20 features
#del df['genres'] 

#df = KEncode(df, 'keywords', 'name', 100) # +10.000 features
del df['keywords'] 

#df = KEncode(df, 'production_countries', 'name', 10) # +100 features
#df = KEncode(df, 'production_companies', 'name', 100) # +5.000 features


#Remove unnecessary columns
del df['id'] 
del df['spoken_languages'] 
del df['production_companies'] 
del df['production_countries'] 
del df['homepage'] 
del df['overview']

#Fill NaNs
df = df.fillna(0)