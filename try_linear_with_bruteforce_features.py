"""

Purpose: This script investigates whether or not synthetically constructed features will help make
the linear model (LogisticRegression) predict better.
Author: Eric Chio "log0" <im.ckieric@gmail.com>

======================================================================================================
Summary:

Our test aims to find out if artificial features could make the prediction more accurate. In our case,
we found that an additional of 2 features or more (despite discovered via bruteforce manner) does
increase the prediction capabilities.

======================================================================================================
Methodology:

The way we will do it is to:
1) Run without any modifications to original dataset
2) Construct a variable from every pair of two features via multiple / divide / add / subtract ,
use the feature one at a time
3) Construct two variables from every pair of three features via multiple / divide / add / subtract ,
use the feature two at a time
4) Construct three variables from every pair of four features via multiple / divide / add / subtract ,
use the feature three at a time
5) Construct n^2 features (n = number of features) for each pair of features via multiple / divide /
add / subtract , and use the features all at the same time

======================================================================================================
Log Output:

1) Run without any modifications to original dataset
Final best score : 0.854368932039
Final best estimator : LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty=l2, random_state=None, tol=0.0001)
2) Construct a variable from every pair of two features via multiple / divide / add / subtract , use the feature one at
a time
Final best score : 0.870550161812
Final best indices : (4, 4)
Final best estimator : LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty=l2, random_state=None, tol=0.0001)
3) Construct two variables from every pair of three features via multiple / divide / add / subtract , use the feature tw
o at a time
Final best score : 0.870550161812
Final best indices : (2, 4, 4)
Final best estimator : LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty=l2, random_state=None, tol=0.0001)
4) Construct three variables from every pair of four features via multiple / divide / add / subtract , use the feature t
hree at a time
Final best score : 0.870550161812
Final best indices : (0, 4, 4, 2)
Final best estimator : LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty=l2, random_state=None, tol=0.0001)
5) Construct n^2 features (n = number of features) for each pair of features via multiple / divide / add / subtract , an
d use the features all at the same time
X.shape = (309L, 42L)
Final best score : 0.844660194175
Final best estimator : LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty=l2, random_state=None, tol=0.0001)

======================================================================================================
Data Set Information:

Biomedical data set built by Dr. Henrique da Mota during a medical residence period in the Group
of Applied Research in Orthopaedics (GARO) of the Centre MÃ©dico-Chirurgical de RÃ©adaptation des
Massues, Lyon, France. The data have been organized in two different but related classification
tasks. The first task consists in classifying patients as belonging to one out of three
categories: Normal (100 patients), Disk Hernia (60 patients) or Spondylolisthesis (150
patients). For the second task, the categories Disk Hernia and Spondylolisthesis were merged 
into a single category labelled as 'abnormal'. Thus, the second task consists in classifying
patients as belonging to one out of two categories: Normal (100 patients) or Abnormal (210 
patients). We provide files also for use within the WEKA environment.

Attribute Information:

Each patient is represented in the data set by six biomechanical attributes derived from the 
shape and orientation of the pelvis and lumbar spine (in this order): pelvic incidence, pelvic
tilt, lumbar lordosis angle, sacral slope, pelvic radius and grade of spondylolisthesis. The
following convention is used for the class labels: DH (Disk Hernia), Spondylolisthesis (SL),
Normal (NO) and Abnormal (AB).

"""
import csv
import random
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import *

def run_cv(X, Y):
    results = {}
    
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    # param_grid = {'C': [1]} # Use this if it runs too slow
    
    clf = LogisticRegression()
    
    grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=0, n_jobs=1)
    grid_search.fit(X, Y)
    
    """
    # Debug text
    print 'Begin grid search...'
    print 'X.shape = %s' % (str(X.shape))
    
    for i in grid_search.grid_scores_:
        print i
    """
    
    results['best_cv_score'] = grid_search.best_score_
    results['best_estimator'] = grid_search.best_estimator_
    
    """
    # Debug text
    print 'Best score : %s' % (results['best_cv_score'])
    print 'Best estimator : %s' % (results['best_estimator'])
    print ''
    """
    
    return results

def build(X, i, j):
    new_x = (X[:, i] * X[:, j])
    new_x = new_x.reshape(new_x.shape[0], 1)
    
    X2 = np.hstack((X, new_x))
    return X2
    
def run_bare(X, Y):
    """
    1) Run without any modifications to original dataset
    """
    
    result = run_cv(X, Y)
    
    print 'Final best score : %s' % result['best_cv_score']
    print 'Final best estimator : %s' % result['best_estimator']

def run_all_features(X, Y):
    """
    2) Construct a variable from every pair of two features via multiple / divide / add / subtract ,
    use the feature one at a time
    """

    scores = []
    estimators = []
    indices = []
    
    X2 = X
    for i in xrange(X.shape[1]):
        for j in xrange(X.shape[1]):
            new_x = (X[:, i] * X[:, j])
            new_x = new_x.reshape(new_x.shape[0], 1)            
            X2 = np.hstack((X2, new_x))
    
    print 'X.shape = %s' % (str(X2.shape))
    
    result = run_cv(X2, Y)
    
    scores.append(result['best_cv_score'])
    estimators.append(result['best_estimator'])
    
    best_score = scores[0]
    best_estimator = estimators[0]
    
    print 'Final best score : %s' % best_score
    print 'Final best estimator : %s' % best_estimator

def run_one_feature(X, Y):
    """
    3) Construct two variables from every pair of three features via multiple / divide / add / subtract ,
    use the feature two at a time
    """
    
    scores = []
    estimators = []
    indices = []
    
    for i in xrange(X.shape[1]):
        for j in xrange(X.shape[1]):
            new_x = (X[:, i] * X[:, j])
            new_x = new_x.reshape(new_x.shape[0], 1)
            
            X2 = np.hstack((X, new_x))
            result = run_cv(X2, Y)
            
            indices.append((i,j))
            scores.append(result['best_cv_score'])
            estimators.append(result['best_estimator'])
    
    best_score = max(scores)
    best_index = np.where(scores == best_score)[0][0]
    best_estimator = estimators[best_index]
    best_indices = indices[best_index]
    
    print 'Final best score : %s' % best_score
    print 'Final best indices : %s' % (str(best_indices))
    print 'Final best estimator : %s' % best_estimator

def run_two_features(X, Y):
    """
    4) Construct three variables from every pair of four features via multiple / divide / add / subtract ,
    use the feature three at a time
    """

    scores = []
    estimators = []
    indices = []
    
    for i in xrange(X.shape[1]):
        for j in xrange(X.shape[1]):
            for k in xrange(X.shape[1]):
                X2 = build(X, i, j)
                X3 = build(X2, j, k)
                
                result = run_cv(X3, Y)
                
                indices.append((i,j,k))
                scores.append(result['best_cv_score'])
                estimators.append(result['best_estimator'])
    
    best_score = max(scores)
    best_index = np.where(scores == best_score)[0][0]
    best_estimator = estimators[best_index]
    best_indices = indices[best_index]
    
    print 'Final best score : %s' % best_score
    print 'Final best indices : %s' % (str(best_indices))
    print 'Final best estimator : %s' % best_estimator
    
def run_three_features(X, Y):
    """
    5) Construct n^2 features (n = number of features) for each pair of features via multiple / divide /
    add / subtract , and use the features all at the same time
    """

    scores = []
    estimators = []
    indices = []
    
    for i in xrange(X.shape[1]):
        for j in xrange(X.shape[1]):
            for k in xrange(X.shape[1]):
                for l in xrange(X.shape[1]):
                    X2 = build(X, i, j)
                    X3 = build(X2, j, k)
                    X4 = build(X3, k, l)
                    
                    result = run_cv(X4, Y)
                    
                    indices.append((i,j,k,l))
                    scores.append(result['best_cv_score'])
                    estimators.append(result['best_estimator'])
    
    best_score = max(scores)
    best_index = np.where(scores == best_score)[0][0]
    best_estimator = estimators[best_index]
    best_indices = indices[best_index]
    
    print 'Final best score : %s' % best_score
    print 'Final best indices : %s' % (str(best_indices))
    print 'Final best estimator : %s' % best_estimator

if __name__ == '__main__':
    SEED = 50091
    random.seed(SEED)
    
    train_file = 'data/column_3C.dat'

    data = [ i for i in csv.reader(file(train_file, 'rb'), delimiter=' ') ]
    data = data[1:] # remove header
    
    X = np.array([ i[:-1] for i in data ], dtype=float)
    Y = np.array([ i[-1] for i in data ])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(Y)
    Y = label_encoder.transform(Y)
    
    print '1) Run without any modifications to original dataset'
    run_bare(X, Y)
    
    print '2) Construct a variable from every pair of two features via multiple / divide / add / subtract , use the feature one at a time'
    run_one_feature(X, Y)
    
    print '3) Construct two variables from every pair of three features via multiple / divide / add / subtract , use the feature two at a time'
    run_two_features(X, Y)
    
    print '4) Construct three variables from every pair of four features via multiple / divide / add / subtract , use the feature three at a time'
    run_three_features(X, Y)
    
    print '5) Construct n^2 features (n = number of features) for each pair of features via multiple / divide / add / subtract , and use the features all at the same time'
    run_all_features(X, Y)