from data_import import *

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Simple scatter plot
plt.plot(df['vote_average'], df['budget'], 'o')
plt.xlabel('vote_average')
plt.ylabel('budget')
plt.title('vote_average vs budget')
plt.show()   

# Simple scatter plot
plt.plot(df['vote_average'], df_pca['release_date'], 'o')
plt.xlabel('vote_average')
plt.ylabel('release_date')
plt.title('vote_average vs release_date')
plt.show()         


# Simple scatter plot
plt.plot(df_pca['release_date'], df['budget'], 'o')
plt.xlabel('release_date')
plt.ylabel('budget')
plt.title('release_date vs budget')
plt.show()         

# Simple scatter plot
plt.plot(df_pca['release_date'], df['budget'], 'o')
plt.xlabel('release_date')
plt.ylabel('budget')
plt.title('release_date vs budget')
plt.show()         