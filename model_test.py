import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df_train = pd.read_pickle('../data/diabetes_train.pickle')
df_validation = pd.read_pickle('../data/diabetes_validation.pickle')
df = np.array(pd.concat([df_train, df_validation]))
X = df[:, 0:-1]
Y = df[:, -1]

config = pickle.load(open( "../model/decision_tree_feature_selection_model.pickle", "rb" ))

max_depth = config['max_depth']
n_estimator = config['n_estimator']

dt = DecisionTreeClassifier(max_depth=max_depth)
dt.fit(X, Y)

test_file = np.array(pd.read_pickle('../data/diabetes_test.pickle'))
tst_X = test_file[:, 0:-1]
tst_Y = test_file[:, -1]

model = SelectFromModel(dt, prefit=True)
X_new = model.transform(tst_X)

clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator)
clf.fit(X_new, tst_Y)
print('accuracy on test set: ', clf.score(X_new, tst_Y))