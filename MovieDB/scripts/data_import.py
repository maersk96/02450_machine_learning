import numpy as np
import pandas as pd

filename = '../data/tmdb_5000_movies.csv'
df = pd.read_csv(filename)

# Summary statistics
df_stats = pd.DataFrame()
df_stats['mean'] = df.mean(axis=0)
df_stats['std'] = df.std(axis=0)
df_stats['median'] = df.median(axis=0)


# Drop nan-values
df = df.drop(4553) # No release data + sad info