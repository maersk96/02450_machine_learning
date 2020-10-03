from data_import import *

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Simple scatter plot
plot(df['vote_count'], df['popularity'], 'o')
