import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def compute(nn, trn_X, trn_Y, tst_X, tst_Y):
    nn.fit(trn_X, trn_Y)
    result = nn.predict(tst_X)

    confusion = [[0, 0], [0, 0]]
    for i in range(len(tst_Y)):
        if tst_Y[i] > 0.0 and result[i] > 0.0:
            confusion[1][1] += 1
        elif tst_Y[i] > 0.0 and result[i] < 1.0:
            confusion[1][0] += 1
        elif tst_Y[i] < 1.0 and result[i] > 0.0:
            confusion[0][1] += 1
        elif tst_Y[i] < 1.0 and result[i] < 1.0:
            confusion[0][0] += 1

    precision = confusion[1][1] / (confusion[0][1] + confusion[1][1])
    sensitivity = confusion[1][1] / (confusion[1][0] + confusion[1][1])
    f_one = 2 * confusion[1][1] / (confusion[1][0] + 2 * confusion[1][1] + confusion[0][1])
    specificity = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    return precision, sensitivity, f_one, specificity

# Train model (PCA)
df = pd.read_pickle('../data/diabetes_train.pickle')
df_validation = pd.read_pickle('../data/diabetes_validation.pickle')
df_train = np.array(pd.concat([df, df_validation]))

trn_X = df_train[:, 0:-1]
trn_Y = df_train[:, -1]

config = pickle.load(open( "../model/decision_tree_feature_selection_model.pickle", "rb" ))

max_depth = config['max_depth']
n_estimator = config['n_estimator']

dt = DecisionTreeClassifier(max_depth=max_depth)
dt.fit(trn_X, trn_Y)

test_file = np.array(pd.read_pickle('../data/diabetes_test.pickle'))
tst_X = test_file[:, 0:-1]
tst_Y = test_file[:, -1]

model = SelectFromModel(dt, prefit=True)
X_new_trn = model.transform(trn_X)
X_new_tst = model.transform(tst_X)

clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator)

prec_list = []
sen_list = []
f_one_list = []
spec_list = []
i_list = []
maxi = 30
for i in range(maxi - 5):
    mean_prec, mean_sen, mean_f_one, mean_spec = 0.0, 0.0, 0.0, 0.0
    print('computing epoch = ', i + 5)
    i_list.append(i + 5)
    for j in range(i + 5):
        precision, sensitivity, f_one, specificity = compute(clf, X_new_trn, trn_Y, X_new_tst, tst_Y)
        mean_prec += precision
        mean_sen += sensitivity
        mean_f_one += f_one
        mean_spec += specificity
    prec_list.append(mean_prec / (i + 5))
    sen_list.append(mean_sen / (i + 5))
    f_one_list.append(mean_f_one / (i + 5))
    spec_list.append(mean_spec / (i + 5))


colors = [(153/255, 77/255, 82/255),
          (36/255, 169/255, 225/255),
          (255/255, 150/255, 128/255),
          (182/255, 194/255, 154/255)]

plt.style.use('ggplot')
plt.xlim(0, maxi + 5)
plt.scatter(i_list, sen_list, label='sensitivity', color=colors[0], s=8)
plt.scatter(i_list, spec_list, label='specificity', color=colors[1], s=8)
plt.scatter(i_list, prec_list, label='precision', color=colors[2], s=8)
plt.scatter(i_list, f_one_list, label='F1 score', color=colors[3], s=8)
plt.legend(loc='lower right')
plt.ylim(0.5, 1.1)
title = 'Mean Sensitivity, Specificity, F1 score and Precision'
plt.title(title)
plt.xlabel('epochs')
plt.ylabel('mean')
plt.savefig('../pic/confusionMatrix.png')
plt.close()