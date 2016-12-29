import numpy as np
import pickle
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import preprocessing


config = pickle.load(open( "../model/decision_tree_feature_selection_configs.pickle", "rb" ))

depth_list = []
n_estimator_list = []
mean_accuracy_list = []
duration_list = []

for conf in config:
    depth_list.append(conf['max_depth'])
    n_estimator_list.append(conf['n_estimator'])
    mean_accuracy_list.append(conf['mean_accuracy'])
    duration_list.append(conf['duration'])

min_max_scaler = preprocessing.MinMaxScaler((1, 20))
mean_accuracy_list = min_max_scaler.fit_transform(np.array(mean_accuracy_list).reshape((-1, 1)))

trace0 = go.Scatter(
    x=depth_list,
    y=n_estimator_list,
    mode='markers',
    marker=dict(
        color=duration_list,
        size=mean_accuracy_list,
        showscale=True
    )
)

data = [trace0]
py.iplot(data, filename='bubblechart1')