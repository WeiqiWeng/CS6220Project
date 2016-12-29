# nominal feature mapping
discharge_list = [x for x in range(1, 30)]
discharge_list.remove(25)
discharge_list.remove(26)
admission_source_list = [x for x in range(1, 27)]
admission_source_list.remove(15)
admission_source_list.remove(17)
admission_source_list.remove(20)
admission_source_list.remove(21)
dosage = ['Up', 'Down', 'Steady', 'None']
diag_id = [x for x in range(9)]

nominal_feature_mapping = {'age': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
                                   '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                           'race': ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'],
                           'admission_type_id': [x for x in range(1, 8)],
                           'discharge_disposition_id': discharge_list,
                           'admission_source_id': admission_source_list,
                           'max_glu_serum': ['>200', '>300', 'Norm', 'None'],
                           'A1Cresult': ['>7', '>8', 'Norm', 'None'],
                           'metformin': dosage,
                           'repaglinide': dosage,
                           'nateglinide': dosage,
                           'chlorpropamide': dosage,
                           'glimepiride': dosage,
                           'acetohexamide': dosage,
                           'glipizide': dosage,
                           'glyburide': dosage,
                           'tolbutamide': dosage,
                           'pioglitazone': dosage,
                           'rosiglitazone': dosage,
                           'acarbose': dosage,
                           'miglitol': dosage,
                           'troglitazone': dosage,
                           'tolazamide': dosage,
                           'examide': dosage,
                           'citoglipton': dosage,
                           'insulin': dosage,
                           'glyburide-metformin': dosage,
                           'glipizide-metformin': dosage,
                           'glimepiride-pioglitazone': dosage,
                           'metformin-rosiglitazone': dosage,
                           'metformin-pioglitazone': dosage,
                           'diag_1': diag_id,
                           'diag_2': diag_id,
                           'diag_3': diag_id
                           }

nominal_features = ['age', 'race', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                    'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                    'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                    'metformin-rosiglitazone', 'metformin-pioglitazone', 'diag_1', 'diag_2', 'diag_3']

numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                   'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']