import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

pc = pickle.load(open("../model/random_forest_config.pickle",'rb'))
n_estimators = pc['n_estimators']
max_depth = pc['max_depth']
df = pd.read_pickle('../data/diabetes_validation.pickle')
y = np.array(df['readmitted'])
del df['readmitted']
x = np.array(df)
folds = range(5,51,5)
mean_score = []
std_score = []
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
for n in folds:
    print("the current folds is: ", n)
    result = cross_val_score(clf, x, y, cv=n)
    mean_score.append(np.mean(result))
    std_score.append(np.std(result))

plt.errorbar(folds,mean_score,yerr = std_score,label = "random forest")
plt.xlim((0,60))
plt.legend(loc='best')
plt.ylabel("mean accuracy with std")
plt.xlabel("number of folds")
title = 'error bar plot'
plt.title(title)
plt.savefig('../pic/random_forest.png')
plt.close()