from data_prepare_classification import *

# exercise 5.1.2
import numpy as np
from sklearn import tree
from platform import system
import os
from os import getcwd
from toolbox_02450 import windows_graphviz_call
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Fit regression tree classifier, Gini split criterion, no pruning
#criterion='gini'
criterion='entropy'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2, min_impurity_decrease=0.000)
dtc = dtc.fit(X,y)

fname='tree_' + criterion
# Export tree graph .gvz file to parse to graphviz
out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)

# Depending on the platform, we handle the file differently, first for Linux 
# Mac
if system() == 'Linux' or system() == 'Darwin':
    import graphviz
    # Make a graphviz object from the file
    src=graphviz.Source.from_file(fname + '.gvz')
    print('\n\n\n To view the tree, write "src" in the command prompt \n\n\n')
    
# ... and then for Windows:
if system() == 'Windows':
    # N.B.: you have to update the path_to_graphviz to reflect the position you 
    # unzipped the software in!
    windows_graphviz_call(fname=fname,
                          cur_dir=getcwd(),
                          path_to_graphviz=r'C:\Program Files\Graphviz 2.44.1')
    plt.figure(figsize=(12,12))
    #plt.imshow(imread(fname + '.png'))
    plt.box('off'); plt.axis('off')
    plt.show()

print('Ran Exercise 5.1.2')

###Test movies
##BORAT SUBSEQUENT MOVIEFILM
#x = np.array([26, 99, 96, 10000000, 1, 10, 2020,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,-1)
##KNIVES OUT
#x = np.array([10, 30, 130, 40000000, 1, 11, 2019,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0]).reshape(1,-1)


# Evaluate the classification tree for the new data object
x_class = dtc.predict(x)[0]

print('\nNew object attributes:')
print(dict(zip(attributeNames,x[0])))
print('\nClassification result:')
print(classNames[x_class])

print('Ran Exercise 5.1.4')

