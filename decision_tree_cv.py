import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.style.use('ggplot')

df = np.array(pd.read_pickle('../data/diabetes_validation.pickle'))
X = df[:, 0:-1]
Y = df[:, -1]

dt_config = pickle.load(open( "../model/decision_tree_model.pickle", "rb" ))
max_depth = dt_config.get_params()['max_depth']
print (max_depth)
dt = DecisionTreeClassifier(max_depth=max_depth)

folds = range(5, 51, 5)

scores_dt = []
std_dt = []
for n in folds:
    print('%d folds for DT' % n)
    result = cross_val_score(dt, X, Y, cv=n)
    scores_dt.append(np.mean(result))
    std_dt.append(np.std(result))

plt.errorbar(folds, scores_dt, yerr=std_dt, label='DT')
plt.xlim((0, 60))
plt.legend(loc='best')
plt.ylabel("mean accuracy with std")
plt.xlabel("number of folds")
title = 'error bar plot'
plt.title(title)
plt.savefig('../pic/errorBarDT.png')
plt.close()
