import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def ran(path):
	df = pd.read_pickle('../data/diabetes_train.pickle')
	y = np.array(df['readmitted'])
	del df['readmitted']
	x = np.array(df)
	para_grid = {'n_estimators':range(50,501,50),
				 'max_depth':range(1,21,1)}
	clf = RandomForestClassifier()
	gs = GridSearchCV(clf,para_grid)
	print(x.shape,y.shape)
	gs.fit(x,y)
	print('best mean CV accuracy = ', gs.best_score_)
	pickle.dump({'n_estimators': gs.best_params_['n_estimators'],
				 'max_depth': gs.best_params_['max_depth']},open(path,'wb'))


def main():
	ran("../model/random_forest_config.pickle")

main()
