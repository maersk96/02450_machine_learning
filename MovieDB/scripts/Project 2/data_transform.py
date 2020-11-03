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
#Json formatting

def KEncode(column, key):
    return df.drop(column, axis=1).join(pd.get_dummies(
            pd.Series(df[column].apply(json.loads).map(lambda x:[i[key] for i in x])
                .apply(pd.Series).stack().reset_index(1, drop=True))).sum(level=0)
           ).fillna(0)
    
df = KEncode('genres', 'name')
#df = KEncode('production_companies', 'name')
#df = KEncode('keywords', 'name')




#TODO
#DONE -- trans_lang();
#trans_genre();
#trans_keywords();
#trans_prodComp();
#DONE -- trans_title();
#DONE -- trans_tagline();
#DONE -- trans_month();
#DONE -- trans_year();
#DONE -- trans_voteAvg();


# General variables
# ...

# Classification variabels
y = df['vote_average'].get_values()
del df['vote_average']