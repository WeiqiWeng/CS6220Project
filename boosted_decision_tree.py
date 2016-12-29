import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier

df = np.array(pd.read_pickle('../data/diabetes_train.pickle'))
X = df[:, 0:-1]
Y = df[:, -1]
print('proportion of readmitted samples in training set: ', len(Y[Y == 1])/len(Y))
max_depth_list = range(1, 21)
base_classifier_list = [DecisionTreeClassifier(max_depth=x) for x in max_depth_list]
param_grid = {'n_estimators': range(50, 501, 50), 'base_estimator': base_classifier_list}
boost_clf = AdaBoostClassifier(algorithm='SAMME')
gs = GridSearchCV(boost_clf, param_grid)
gs.fit(X, Y)
param = gs.best_params_
print('on training set, best mean CV accuracy = ', gs.best_score_)
print('achieved with base decision tree max depth = %d, number of estimator = %d' %
      (param['base_estimator'].get_params()['max_depth'], param['n_estimators']))
pickle.dump(gs.best_estimator_, open("../model/boosted_decision_tree_model.pickle", "wb"))
