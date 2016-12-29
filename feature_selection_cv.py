import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
plt.style.use('ggplot')


df = np.array(pd.read_pickle('../data/diabetes_validation.pickle'))

X = df[:, 0:-1]
Y = df[:, -1]

config = pickle.load(open( "../model/decision_tree_feature_selection_model.pickle", "rb" ))
transformed_X = config['transformed_X']
ori_X = np.array(pd.read_pickle('../data/diabetes_train.pickle'))

max_depth = config['max_depth']
n_estimator = config['n_estimator']
new_feature_size = config['transformed_X'].shape[1]

dt = DecisionTreeClassifier(max_depth=max_depth)
dt.fit(X, Y)

model = SelectFromModel(dt, prefit=True)
X_new = model.transform(X)

# while X_new.shape[1] != new_feature_size:
#     dt = DecisionTreeClassifier(max_depth=max_depth)
#     dt.fit(X, Y)
#
#     model = SelectFromModel(dt, prefit=True)
#     X_new = model.transform(X)

clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator)
# clf.fit(X_new, Y)
# print(clf.score(X_new, Y))
folds = range(5, 51, 5)
scores = []
std = []
for n in folds:
    print('%d folds for model' % n)
    result = cross_val_score(clf, X_new, Y, cv=n)
    scores.append(np.mean(result))
    std.append(np.std(result))

plt.errorbar(folds, scores, yerr=std, label='Feature-selected Random Forest')
plt.xlim((0, 60))
plt.legend(loc='best')
plt.ylabel("mean accuracy with std")
plt.xlabel("number of folds")
title = 'error bar plot'
plt.title(title)
plt.savefig('../pic/errorBarFeatureSelected.png')
plt.close()

