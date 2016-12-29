import data_preprocessing_model as dp
import constants
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek


diabetic_df = pd.read_csv('../data/diabetic_data.csv')
# diabetic_df.to_pickle('../data/diabetic_data.pickle')
#
# # load data from pickle
# diabetic_df = pd.read_pickle('../data/diabetic_data.pickle')

# drop features with too much missing data
diabetic_df.drop(['encounter_id', 'weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

# transform label into binary
diabetic_df.loc[(diabetic_df.readmitted == 'NO') | (diabetic_df.readmitted == '>30'), 'readmitted'] = 0
diabetic_df.loc[diabetic_df.readmitted == '<30', 'readmitted'] = 1

print_format = 'total number of class 0: %d, total number of class 1: %d'
print(print_format %
      (len(diabetic_df[diabetic_df.readmitted == 0]), len(diabetic_df[diabetic_df.readmitted == 1])))

# drop missing record
# 3 dropped
diabetic_df.drop(diabetic_df[diabetic_df.gender == 'Unknown/Invalid'].index, inplace=True)


# diabetic_df.drop(diabetic_df[(diabetic_df.diag_1 == '?') |
#                              (diabetic_df.diag_2 == '?') |
#                              (diabetic_df.diag_3 == '?')].index, inplace=True)

# drop replicated record for a single patient
diabetic_df.drop_duplicates(subset='patient_nbr', keep='first', inplace=True)
diabetic_df.drop(['patient_nbr'], axis=1, inplace=True)

# generate a temporary column for the convenience of assigning binary features
diabetic_df['tmp_ones'] = np.ones((len(diabetic_df), 1))

# transform several feature into binary
diabetic_df.loc[diabetic_df.gender == 'Male', 'gender'] = 1
diabetic_df.loc[diabetic_df.gender == 'Female', 'gender'] = 0

diabetic_df.loc[diabetic_df.change == 'Ch', 'change'] = 1
diabetic_df.loc[diabetic_df.change == 'No', 'change'] = 0

diabetic_df.loc[diabetic_df.diabetesMed == 'Yes', 'diabetesMed'] = 1
diabetic_df.loc[diabetic_df.diabetesMed == 'No', 'diabetesMed'] = 0

# merge record with race == '?' into race == 'Other'
diabetic_df.loc[diabetic_df.race == '?', 'race'] = 'Other'

# merge record with admission_type_id == 'Not Mapped' into admission_type_id == 'NULL'
diabetic_df.loc[diabetic_df.admission_type_id == 8, 'admission_type_id'] = 6

# merge record with discharge_disposition_id == 'Not Mapped' or 'Unknown/Invalid'
# into discharge_disposition_id == 'NULL'
diabetic_df.loc[(diabetic_df.discharge_disposition_id == 25) |
                (diabetic_df.discharge_disposition_id == 26), 'discharge_disposition_id'] = 18

# merge record with admission_source_id == 'NULL' or 'Unknown/Invalid' or 'Not Mapped'
# into admission_source_id == 'Not Available'
diabetic_df.loc[(diabetic_df.admission_source_id == 21) |
                (diabetic_df.admission_source_id == 20) |
                (diabetic_df.admission_source_id == 17) |
                (diabetic_df.admission_source_id == 15), 'admission_source_id'] = 9

# numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
#                    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

diabetic_df[constants.numeric_features] = preprocessing.scale(diabetic_df[constants.numeric_features])

diag_features = ['diag_1', 'diag_2', 'diag_3']

for diag_id in diag_features:
    diabetic_df[diag_id] = pd.to_numeric(diabetic_df[diag_id], errors='coerce')

diabetic_df.fillna(0, inplace=True)
# print(diabetic_df[diag_features].describe())

# diabetic_df.to_csv('test.csv')
diabetic_df = dp.group_diagnosis_id(diabetic_df, diag_features)

# nominal_features = ['age', 'race', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
#                     'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
#                     'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
#                     'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
#                     'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
#                     'metformin-rosiglitazone', 'metformin-pioglitazone', 'diag_1', 'diag_2', 'diag_3']

diabetic_df = dp.bash_add_dummy_indicator(diabetic_df, constants.nominal_features)

diabetic_df.drop(['tmp_ones'], axis=1, inplace=True)


diabetic_label = diabetic_df.readmitted
diabetic_df.drop(['readmitted'], axis=1, inplace=True)

# fix imbalance
sm = SMOTETomek(ratio=0.7)
resample_x, resample_y = sm.fit_sample(diabetic_df, diabetic_label)
df = pd.DataFrame()
features = diabetic_df.columns.values
for i in range(len(features)):
    df[features[i]] = resample_x[:, i]

features = df.columns.values.tolist()
for i in constants.numeric_features:
    features.remove(i)
for i in features:
    df[i] = round(df[i])

print('after fixing imbalance, sample size = ', df.shape[0])

df['readmitted'] = resample_y

print(print_format %
      (len(df[df.readmitted == 0]), len(df[df.readmitted == 1])))

train, test = train_test_split(df, test_size = 0.4)
print('training data set size: ',len(train))
validation, test = train_test_split(test, test_size = 0.5)
print('validation data set size: ',len(validation))
print('test data set size: ',len(test))

train.to_pickle('../data/diabetes_train.pickle')
validation.to_pickle('../data/diabetes_validation.pickle')
test.to_pickle('../data/diabetes_test.pickle')

















