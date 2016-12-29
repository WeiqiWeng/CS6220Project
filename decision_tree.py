import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


df = np.array(pd.read_pickle('../data/diabetes_train.pickle'))

X = df[:, 0:-1]
Y = df[:, -1]

param_grid = {'max_depth': range(1, 21)}
dt = DecisionTreeClassifier()

gs = GridSearchCV(dt, param_grid)
gs.fit(X, Y)

print('best mean CV accuracy = ', gs.best_score_)
print('max depth = ', gs.best_params_['max_depth'])
pickle.dump(gs.best_estimator_, open("../model/decision_tree_model.pickle", "wb"))
