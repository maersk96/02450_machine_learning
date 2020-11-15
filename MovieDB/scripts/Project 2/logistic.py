import sklearn.linear_model as lm

from data_prepare_logistic import *

regularization_strength = 1e2

mdl = lm.LogisticRegression(solver='sag', multi_class='multinomial', 
                                   tol=1e-4, random_state=1, 
                                   penalty='l2', C=1/regularization_strength)
mdl = mdl.fit(X,y)

weights = mdl.coef_[0]

# Predict
# mdl.predict(X[5].reshape(1,-1))

print(weights)

