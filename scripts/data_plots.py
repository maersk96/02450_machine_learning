from data_import import *

import matplotlib.pyplot as plt
import scipy.stats

min_age = 10
max_age = 35
mean = np.mean(X[:, 1])
std = np.std(X[:, 1])
plt.hist(df_students['age'], color = 'blue', edgecolor = 'black',
         bins = 100, density=True, range=(min_age, max_age))
x = np.linspace(min_age, max_age, 1000)
pdf = scipy.stats.norm.pdf(x,loc=mean,scale=std)
plt.plot(x,pdf,'.',color='orange', linewidth=15.0)

plt.title('Current bachelors age distribution')
plt.xlabel('age')
plt.ylabel('density')
plt.show()