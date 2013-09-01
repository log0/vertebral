"""

Purpose: This script investigates whether or not synthetically constructed features will help make
the linear model (LogisticRegression) predict better.

Author: Eric Chio "log0" <im.ckieric@gmail.com>

======================================================================================================
Summary:

Adding a new synthetic feature which is the harversine distance helps the classifier to achieve
perfect score compared to the other which achieved only mediocre score. We conclude that adding
synthetic features do aid in the classifier's performance.

======================================================================================================
Methodology:

We will generate a dataset consisting of two pairs of latitudes and longitudes. This dataset is
replicated plus with an additional feature of adding a harversine distance between the two pairs
of latitude and longitude points.

We will observe if adding this new synthetic feature will help the classifier to perform better.

======================================================================================================
Log Output:

Best cross validation score without synthetic feature : 0.714285714286
Best cross validation score with synthetic feature : 1.0

======================================================================================================
Data Set Information:

Just a latitude and longitude manually written.

"""
import math
import random
import itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score

def harversine_distance(lat1, lon1, lat2, lon2):
    """
    Formula used to determine the distance between two pairs of latitude and longitude points.
    http://en.wikipedia.org/wiki/Haversine_formula
    """
    radius = 6371 # km
    
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d
    
def generate_dataset(with_synthetic_feature = False):
    """
    We will generate a dataset with each row consisting of two pair of latitude
    and longitude points. If the "with_synthetic_feature" flag is set to True,
    we will also append the harversine distance as a synthetic feature to feed into
    the classifier for training and prediction.
    """
    feasible_threshold = 100
    
    latlons = np.array([
        # These four should be feasible with each others
        [36.9, 109.351831],
        [36.9, 109.3519211],
        [36.94512, 109.36122],
        [36.4, 108.9121031],
        # This one should be infeasible with any others
        [34.4, 110.0],
        # These three should be feasible with each others
        [51.9414, 72.2235],
        [51.9414, 72.2335],
        [52.0434, 72.2335],
    ])
    
    # pairwise but drop all points if they are equal
    X = list([i[0], i[1], j[0], j[1]] for i, j in itertools.product(latlons, latlons) if sum(i) != sum(j))
    Y = []
    
    for i, ((lat1, lon1, lat2, lon2)) in enumerate(X):
        distance = harversine_distance(lat1, lon1, lat2, lon2)
        feasible = distance < feasible_threshold
        
        # print '%s %s , %s %s , %s , %s' % (lat1, lon1, lat2, lon2, distance, feasible)
        
        if with_synthetic_feature:
            X[i].append(distance)
        
        if feasible:
            Y.append(1)
        else:
            Y.append(0)
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

if __name__ == '__main__':
    SEED = 9551
    random.seed(SEED)
    
    clf = LogisticRegression()
    
    X1, Y1 = generate_dataset(False)
    cv_score = cross_val_score(clf, X1, Y1, cv=4)
    print 'Best cross validation score without synthetic feature : %s' % (np.max(cv_score))
    
    X2, Y2 = generate_dataset(True)    
    cv_score = cross_val_score(clf, X2, Y2, cv=4)
    print 'Best cross validation score with synthetic feature : %s' % (np.max(cv_score))