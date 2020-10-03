from data_import import *
from pca_preparation import *

import matplotlib.pyplot as plt
import scipy.stats

attribute = 'revenue'
min_age = df[attribute].min()
max_age = 100000000 #df[attribute].max()
mean = df[attribute].mean()
std = df[attribute].std()
plt.hist(df[attribute], color = 'blue', edgecolor = 'black',
         bins = 50, density=True, range=(min_age, max_age))
x = np.linspace(min_age, max_age, 1000)
pdf = scipy.stats.norm.pdf(x,loc=mean,scale=std)
plt.plot(x,pdf,'.',color='orange', linewidth=15.0)

plt.title('Distribution of ' + attribute)
plt.xlabel(attribute)
plt.ylabel('density')
plt.show()


attribute = 'release_date'
min_age = 0
max_age = df_pca[attribute].max()
mean = df_pca[attribute].mean()
std = df_pca[attribute].std()
plt.hist(df_pca[attribute], color = 'blue', edgecolor = 'black',
         bins = 50, density=True, range=(min_age, max_age))
x = np.linspace(min_age, max_age, 1000)
pdf = scipy.stats.norm.pdf(x,loc=mean,scale=std)
plt.plot(x,pdf,'.',color='orange', linewidth=15.0)

plt.title('release_date distribution')
plt.xlabel(attribute)
plt.ylabel('density')
plt.show()





nrows=3
ncols=2
i=0
for k in [0,3,4,6,7,8]:
    attribute = df.columns[k]
    min_age = df[attribute].min()
    max_age = df[attribute].max()
    mean = df[attribute].mean()
    std = df[attribute].std()
    plt.hist(df[attribute], color = 'blue', edgecolor = 'black',
             bins = 50, density=True, range=(min_age, max_age))
    x = np.linspace(min_age, max_age, 1000)
    pdf = scipy.stats.norm.pdf(x,loc=mean,scale=std)
    plt.plot(x,pdf,'.',color='orange', linewidth=15.0)
    
    plt.title('Distribution of ' + attribute)
    plt.xlabel(attribute)
    plt.ylabel('density')
    plt.show()
    i += 1