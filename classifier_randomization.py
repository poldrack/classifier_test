# -*- coding: utf-8 -*-
"""
Example of how to use randomization to assess potential errors
in classification
"""

import numpy,pandas
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import ttest_1samp

def generate_data(nfeatures=10,nsamples=100,noise_sd=1.0):

    X=numpy.random.randn(nsamples, nfeatures)
    
    betas=numpy.random.randn(nfeatures)
    
    y = X.dot(betas) + numpy.random.randn(nsamples)*noise_sd
    y= ((y - numpy.mean(y)) > 0).astype('int')  # ensure equal sized classes
    
    return X,y
    
def cheating_classifier(X,y):
    """
    this is a function that implements a classification analysis
    in which there is peeking at the test data due to an error
    """
    skf=StratifiedKFold(y,n_folds=4)
    pred=numpy.zeros(len(y))
    knn=KNeighborsClassifier()
    for train,test in skf:
        knn.fit(X,y) # this is using the entire dataset, rather than just training
        pred[test]=knn.predict(X[test,:])
    return numpy.mean(pred==y)
    
def shuffle_test(X,y,clf,nperms=1000):
    acc=[]
    y_shuf=y.copy()
    
    for i in range(nperms):
        numpy.random.shuffle(y_shuf)
        acc.append(clf(X,y_shuf))
    return acc

def crossvalidated_classifier(X,y):
    skf=StratifiedKFold(y,n_folds=4)
    pred=numpy.zeros(len(y))
    knn=KNeighborsClassifier() 
    for train,test in skf:
        knn.fit(X[train,:],y[train])
        pred[test]=knn.predict(X[test,:])
    return numpy.mean(pred==y)
        
    
    
if __name__=='__main__':
    X,y=generate_data(noise_sd=1)
    df=pandas.DataFrame(columns=['cheat','cv'])
    
    cheat_noshuf=cheating_classifier(X,y)
    cv_noshuf=crossvalidated_classifier(X,y)
    df.cheat=shuffle_test(X,y,cheating_classifier)
    df.cv=shuffle_test(X,y,crossvalidated_classifier)
    
    print('Accuracy values on original dataset:')
    print('Peeking at test: %f'%numpy.mean(cheat_noshuf))
    print('No peeking (crossvalidation): %f'%numpy.mean(cv_noshuf))
    print('Accuracy with random shuffling of y variable')
    print('Peeking at test: %f'%df.cheat.mean())
    print('No peeking (crossvalidation): %f'%df.cv.mean())
