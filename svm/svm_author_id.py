#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
from sklearn import svm

sys.path.append("../tools/")
from collections import Counter
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### make sure you use // when dividing for integer division

#########################################################
### your code goes here ###
# These lines effectively slice the training dataset down to 1% of its original size, tossing out 99% of the training data.
# features_train = features_train[:len(features_train)//100]
# labels_train = labels_train[:len(labels_train)//100]

t0 = time()
clf = svm.SVC(kernel='rbf', C=10000)
svm_classfier = clf.fit(features_train, labels_train)
print("training time = ", round(time()-t0), "s")

t1 = time()
pred = svm_classfier.predict(features_test)
print("predication is :", pred)
print("predication time = ", round(time()-t1), "s")

accuarcy = clf.score(features_test, labels_test)
print("Sum Predication", sum(pred))
counter = Counter(pred)
print("'Chris 1's: ", counter[1])

print("Predication", pred[10], pred[26], pred[50])

print(accuarcy)
#########################################################
