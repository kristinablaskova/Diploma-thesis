import pandas as pd
import numpy as np
import os
import sklearn.model_selection as ms
import data_preprocessing as dp
import seaborn as sns
import matplotlib.pyplot as plt



local_data = '/Users/kristina/PycharmProjects/vyskumak/Data/'
local_list = '/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv'


def preprocess_data(data):

    n_features = data.shape[1] - 2
    train, test = ms.train_test_split(data, test_size=0.0, shuffle=False)
    data_columns = list(data.columns.values)
    del data
    hidden_sequence = train['hypnogram_User'].tolist()
    l = len(hidden_sequence)
    for i in reversed(range(0, l)):
        if hidden_sequence[i] == "NotScored":
            train = train.drop([i])
            del hidden_sequence[i]
    train = train.drop(['hypnogram_Machine'], axis=1)
    test = test.drop(['hypnogram_Machine'], axis=1)

    observation_sequence = train.iloc[:, 0:n_features].values.tolist()
    return data_columns, hidden_sequence, observation_sequence, train, test

def prepare_training(list_of_training_patients):
    # creates a giant hidden sequence array and train_df from all training patients

    # create array of hidden state sequence of traning dataset
    training_class_array = []

    # create df of observation sequence of traning dataset
    train_df = pd.DataFrame()

    for j in range(list_of_training_patients.shape[0]):


        path = local_data + str(list_of_training_patients['file_name'][j])

        if os.path.isfile(path) == True:



            data = dp.data_import(path)
            binary_features = ["Gain", "Bradycardia", "LegMovement", "CentralApnea", "Arousal", "Hypopnea",
                               "RelativeDesaturation", "Snore", "ObstructiveApnea", "MixedApnea", "LongRR", "Tachycardia"]
            for feature in binary_features:
                if feature in data.columns:
                    data = data.drop(feature, axis=1)
            df1 = data.pop('hypnogram_User')
            data['hypnogram_User'] = df1
            data_columns, hidden_sequence, observation_sequence, train, test = preprocess_data(data=data)
            training_class_array.append(hidden_sequence)
            train_df = train_df.append(train)
        else:
            print('File not found.')
            pass

    feature_names = data.drop(['hypnogram_User', 'hypnogram_Machine'], axis=1).columns.values.tolist()
    del data, observation_sequence, test


    return train_df, training_class_array, feature_names

list_of_patients = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv')
list_of_forbidden = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/forbidden_pacients_length_of_wake.csv')

for patient in list_of_forbidden['Pacient']:
    list_of_patients = list_of_patients[list_of_patients['file_name'] != patient + str('.csv')]

list_of_patients=list_of_patients.reset_index()

train_df, training_class_array, feature_names = prepare_training(list_of_patients[list_of_patients.index == 0])


nonrem1 = train_df[train_df['hypnogram_User'] == 'NonREM1']
nonrem1 = nonrem1.drop(['hypnogram_User'],axis=1)
nonrem1 = nonrem1.apply(pd.to_numeric)
nonrem2 = train_df[train_df['hypnogram_User'] == 'NonREM2']
nonrem2 = nonrem2.drop(['hypnogram_User'],axis=1)
nonrem2 = nonrem2.apply(pd.to_numeric)
nonrem3 = train_df[train_df['hypnogram_User'] == 'NonREM3']
nonrem3 = nonrem3.drop(['hypnogram_User'],axis=1)
nonrem3 = nonrem3.apply(pd.to_numeric)
rem = train_df[train_df['hypnogram_User'] == 'REM']
rem = rem.drop(['hypnogram_User'],axis=1)
rem = rem.apply(pd.to_numeric)
wake = train_df[train_df['hypnogram_User'] == 'Wake']
wake = wake.drop(['hypnogram_User'],axis=1)
wake = wake.apply(pd.to_numeric)

sns.distplot(wake['EEG_C4_A1: GAMMA'],kde = True,color='red',norm_hist=True,kde_kws={"label": "Wake"})
sns.distplot(nonrem1['EEG_C4_A1: GAMMA'],kde = True,color='b',norm_hist=True,kde_kws={"label": "NonREM1"})
sns.distplot(nonrem2['EEG_C4_A1: GAMMA'],kde = True,color='g',norm_hist=True,kde_kws={"label": "NonREM2"})
sns.distplot(nonrem3['EEG_C4_A1: GAMMA'],kde = True,color='yellow',norm_hist=True,kde_kws={"label": "NonREM3"})
sns.distplot(rem['EEG_C4_A1: GAMMA'],kde = True,color='brown',norm_hist=True,kde_kws={"label": "REM"})
plt.title("Gaussian KDE for emission probabilities")
plt.ylabel('Probability')
