import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = np.array(pd.read_pickle('../data/diabetes_validation.pickle'))
X = df[:, 0:-1]
Y = df[:, -1]

folds = range(5, 51, 5)
nn = pickle.load(open( "../model/boosted_decision_tree_model.pickle", "rb" ))
scores_pca = []
std_pca = []
for n in folds:   
    print('%d folds for AdaBoosted DT' % n)
    result = cross_val_score(nn, X, Y, cv=n)    
    scores_pca.append(np.mean(result))    
    std_pca.append(np.std(result))
plt.errorbar(folds, scores_pca, yerr=std_pca, label='AdaBoosted DT')
plt.xlim((0, 60))
plt.legend(loc='best')
plt.ylabel("mean accuracy with std")
plt.xlabel("number of folds")
title = 'error bar plot'
plt.title(title)
plt.savefig('../pic/errorBarBoostDT.png')
plt.close()
