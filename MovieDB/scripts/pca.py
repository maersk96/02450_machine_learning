from pca_preparation import *

import matplotlib.pyplot as plt
from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# Standardized
Y2 = X - np.ones((N, 1))*X.mean(0)
Y2 = Y2*(1/np.std(Y2,0))

# PCA by computing SVD of Y
U,S,V = svd(Y2,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


# Scree plot
labels = ['PC' + str(x) for x in range(1,len(rho)+1)]
plt.bar(x=range(1,len(rho)+1), height=rho, tick_label=labels)
plt.ylabel('Percentage of explained variace')
plt.xlabel('Principal Component')
plt.title('Scree plot')
plt.show()



plt.plot(V, df_pca['budget'])