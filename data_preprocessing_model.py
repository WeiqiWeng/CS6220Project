import constants
import pandas as pd
import numpy as np
from sklearn import preprocessing


def add_dummy_indicator(data_frame, indicator):
    # item_list = data_frame[indicator].unique()
    # print(item_list)
    for item in constants.nominal_feature_mapping[indicator][0:-1]:
        dummy_ind_name = str(indicator) + "_" + str(item)
        data_frame[dummy_ind_name] = data_frame['tmp_ones'].where(data_frame[indicator] == item, other=0)

    data_frame.drop([indicator], axis=1, inplace=True)
    return data_frame


# def bash_zscore_scale(data_frame, numeric_features):
#
#     for feature in numeric_features:
#         data_frame[feature] = preprocessing.scale(data_frame[feature])
#
#     return data_frame


def bash_add_dummy_indicator(data_frame, nominal_features):

    for feature in nominal_features:
        data_frame = add_dummy_indicator(data_frame, feature)

    return data_frame


def group_diagnosis_id(data_frame, diag_id):

    for id in diag_id:
        # Circulatory: 1
        data_frame.loc[(data_frame[id] >= 390) & (data_frame[id] <= 459), id] = 1
        data_frame.loc[(np.abs(data_frame[id] - 785) < 0.001), id] = 1
        # Respiratory: 2
        data_frame.loc[(data_frame[id] >= 460) & (data_frame[id] <= 519), id] = 2
        data_frame.loc[(data_frame[id] == 786), id] = 2
        # Digestive: 3
        data_frame.loc[(data_frame[id] >= 520) & (data_frame[id] <= 579), id] = 3
        data_frame.loc[(data_frame[id] == 787), id] = 3
        # Diabetes: 4
        data_frame.loc[(data_frame[id] >= 250) & (data_frame[id] < 251), id] = 4
        # Injury: 5
        data_frame.loc[(data_frame[id] >= 800) & (data_frame[id] <= 999), id] = 5
        # Musculoskeletal: 6
        data_frame.loc[(data_frame[id] >= 710) & (data_frame[id] <= 739), id] = 6
        # Genitourinary: 7
        data_frame.loc[(data_frame[id] >= 580) & (data_frame[id] <= 629), id] = 7
        data_frame.loc[(data_frame[id] == 788), id] = 7
        # Neoplasms: 8
        data_frame.loc[(data_frame[id] >= 140) & (data_frame[id] <= 239), id] = 8
        # else: 0
        data_frame.loc[(data_frame[id] != 1) &
                       (data_frame[id] != 2) &
                       (data_frame[id] != 3) &
                       (data_frame[id] != 4) &
                       (data_frame[id] != 5) &
                       (data_frame[id] != 6) &
                       (data_frame[id] != 7) &
                       (data_frame[id] != 8), id] = 0

    return data_frame