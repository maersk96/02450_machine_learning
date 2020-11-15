import sklearn.linear_model as lm

regularization_strength = 1e-3
def RegLog (X_train, y_train):
    #Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
    mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                                   tol=1e-4, random_state=1, 
                                   penalty='l2', C=1/regularization_strength)
    return mdl.fit(X_train,y_train)