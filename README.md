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