# -*- coding: utf-8 -*-
from data_import import *

# Extract program names to python list,
# then encode with integers (dict)
classLabels = df_students['programme']
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(np.unique(classLabels)))))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

#Remove programme & student id
df_students = df_students.drop(["programme","student_id"], axis=1)

# Extract matrix
X = df_students.get_values()
attributeNames = df_students.columns
N = len(X)
M = len(attributeNames)
C = len(classNames)

