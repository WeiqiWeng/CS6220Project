import pandas as pd
import numpy as np
import pickle
import time
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = np.array(pd.read_pickle('../data/diabetes_train.pickle'))

X = df[:, 0:-1]
Y = df[:, -1]

opt_depth = -1
opt_n_estimator = -1
opt_score = -1
transformed_X = np.array([])
config = []

for depth in range(1, 21):
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X, Y)

    model = SelectFromModel(dt, prefit=True)
    X_new = model.transform(X)

    for n in range(50, 501, 50):
        print('computing max depth = %d, number of estimators = %d' % (depth, n))
        clf = RandomForestClassifier(max_depth=depth, n_estimators=n)
        start = time.clock()
        result = cross_val_score(clf, X_new, Y, cv=5)
        end = time.clock()
        duration = end - start
        mean_accuracy = np.mean(result)

        config.append({'max_depth': depth, 'n_estimator': n, 'mean_accuracy': mean_accuracy, 'duration': duration})

        if mean_accuracy > opt_score:
            opt_score = mean_accuracy
            opt_depth = depth
            opt_n_estimator = n
            transformed_X = X_new

print('%d features are selected.' % transformed_X)
print('best mean accuracy = %d, max depth = %d, forest size = %d' % (opt_score, opt_depth, opt_n_estimator))
opt_config = {'max_depth': opt_depth, 'n_estimator': opt_n_estimator, 'transformed_X': transformed_X}

pickle.dump(opt_config, open("../model/decision_tree_feature_selection_model.pickle", "wb"))
pickle.dump(config, open("../model/decision_tree_feature_selection_configs.pickle", "wb"))