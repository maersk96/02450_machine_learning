from data_import import *
import json

#Transform Language - Boolean if(eng) = true else false
df['original_language'] = df['original_language'] == 'en'; 

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
df['genres'] = df['genres'].apply(json.loads);
df['keywords'] = df['keywords'].apply(json.loads);
df['production_companies'] = df['production_companies'].apply(json.loads);

##This does not work!
# df=(df.drop('genres', axis=1)
#     .join(pd.get_dummies(pd.DataFrame.from_records(df['genres'][0])['name'].explode())
#         .sum(level=0)
#     ).fillna(0)
# )


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
