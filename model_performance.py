import pandas as pd
import numpy as np
import pickle
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

model = SelectFromModel(dt, prefit=True)
X_new = model.transform(X)

epoch_list = range(5, 34)
mean_accuracy_list = []
clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator)

for epoch in epoch_list:
    mean_accuracy = 0.0
    for round in range(epoch):
        print('epoch = %d, round %d' % (epoch, round))
        clf.fit(X_new, Y)
        mean_accuracy += clf.score(X_new, Y)

    mean_accuracy_list.append(mean_accuracy/epoch)

plt.scatter(epoch_list, mean_accuracy_list, label='mean accuracy')
plt.legend(loc='best')
plt.ylabel("mean accuracy")
plt.xlabel("epochs")
title = 'mean accuracy'
plt.title(title)
plt.savefig('../pic/meanAccuracy.png')
plt.close()
