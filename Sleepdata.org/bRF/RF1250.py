import pandas as pd
import sklearn.model_selection as ms
import os
import data_preprocessing_ks as dp
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import classification_report,accuracy_score
from itertools import groupby


wake_lengths=[]
od=1200
do=1250

place = 'server'

local_data = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/shhs1/'
local_list = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/list_of_patients.csv'
local_cfeatures = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/RF_balanced/common_features.txt'


server_data = "/disk/bios/sleepdata.org/shhs/polysomnography/csv/shhs1/"
server_list = "/disk/bios/blaskova/kod_sleep_org/list_of_patients.csv"
server_cfeatures = '/disk/bios/blaskova/kod_sleep_org/results/common_features.txt'

if place == 'local':

    cfeatures = local_cfeatures
    data_path = local_data
    plist = local_list
    forbidden_ps1 = pd.read_csv(
        '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/forbidden_pacients_length_of_wake.csv')
    precision_tables = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/RF_balanced/rf_pt.txt', 'w')
    confusion_matrices = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/RF_balanced/rf_cms.txt', 'w')
    scores_file = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/RF_balanced/rf_scores.txt', 'w')

else:
    cfeatures = server_cfeatures
    data_path = server_data
    plist = server_list
    forbidden_ps1 = pd.read_csv(
        '/disk/bios/blaskova/kod_sleep_org/results/forbidden_pacients_length_of_wake.csv')

    precision_tables = open('/disk/bios/blaskova/kod_sleep_org/RF_balanced/rf_pt' + str(do) + '.txt', 'w')
    confusion_matrices = open('/disk/bios/blaskova/kod_sleep_org/RF_balanced/rf_cms' + str(do) + '.txt', 'w')
    scores_file = open('/disk/bios/blaskova/kod_sleep_org/RF_balanced/rf_scores' + str(do) + '.txt', 'w')

states = ['NonREM1', "NonREM2", "NonREM3", "REM", "Wake"]
confs = np.zeros((len(states), len(states)))
cms=[]
pts=[]
scores=[]

common_features = pd.read_csv(cfeatures)
common_features=[common_features.columns[i].replace(" '",'').replace("['",'').replace("']",'').replace("'",'') for i in range(len(common_features.columns))]



########################################################################################################################
#FUNKCIE
########################################################################################################################

def preprocess_data(data):

    n_features = data.shape[1] - 1
    train, test = ms.train_test_split(data, test_size=0.0, shuffle=False)
    data_columns = list(data.columns.values)
    hidden_sequence = train['staging'].tolist()
    l = len(hidden_sequence)
    for i in reversed(range(0, l)):
        if hidden_sequence[i] == "NotScored":
            train = train.drop([i])
            del hidden_sequence[i]


    observation_sequence = train.iloc[:, 0:n_features].values.tolist()
    return data_columns, hidden_sequence, observation_sequence, train, test

def prepare_training(list_of_training_patients,common_features):
    # creates a giant hidden sequence array and train_df from all training patients

    # create array of hidden state sequence of traning dataset
    training_class_array = []

    # create df of observation sequence of traning dataset
    train_df = pd.DataFrame()

    for j in range(list_of_training_patients.shape[0]):


        path = data_path + str(list_of_training_patients['file_name'][j])

        if os.path.isfile(path) == True:


            data = dp.data_import(path, common_features)
            binary_features = ['spindlesA', 'spindlesB', 'Hypopnea', 'SpO2 desaturation', 'Arousal ()',
                               'Obstructive Apnea', 'SpO2 artifact','Central Apnea']
            for feature in binary_features:
                if feature in data.columns:
                    data = data.drop(feature, axis=1)
            df1 = data.pop('staging')
            data['staging'] = df1
            data_columns, hidden_sequence, observation_sequence, train, test = preprocess_data(data=data)
            training_class_array.append(hidden_sequence)
            train_df = train_df.append(train)
        else:
            print('File not found.')
            pass

    feature_names = data.drop(['staging'], axis=1).columns.values.tolist()
    del data, observation_sequence, test


    return train_df, training_class_array, feature_names

########################################################################################################################


#LOAD PATIENT LIST AND DROP OUTLIERS
list_of_patients = pd.read_csv(plist)
forbidden_ps1['file_name'] = forbidden_ps1['Pacient'] + '.csv'
list_of_patients = list_of_patients[~list_of_patients['file_name'].isin(forbidden_ps1['file_name'])]

print("List of patients loaded")
#list_of_patients.shape[0]
for i in range(od, do):
    print("Predicting the patient number " + str(i))

    # * prepare traning and testing datasets *

    list_of_testing_patients = list_of_patients.iloc[[i]]
    list_of_training_patients = list_of_patients.drop(list_of_patients.index[i], axis=0)

    list_of_training_patients = list_of_training_patients.reset_index()

    train_df, training_class_array, feature_names = prepare_training(list_of_training_patients,common_features)

    y = train_df['staging'].values.tolist()
    X = train_df.values[:,:-1]

    classifier = RandomForestClassifier(n_estimators=10, max_depth=10, class_weight='balanced')
    classifier.fit(X, y)

    list_of_testing_patients = list_of_testing_patients.reset_index()

    for k in range(list_of_testing_patients.shape[0]):
        print(list_of_testing_patients['file_name'])
        print(str(list_of_testing_patients['file_name'][k]))
        print(data_path)
        path = data_path + str(list_of_testing_patients['file_name'][k])

        if os.path.isfile(path) == True:
            patient_data = dp.data_import(path, common_features)
            binary_features = ['spindlesA', 'spindlesB', 'Hypopnea', 'SpO2 desaturation', 'Arousal ()',
                               'Obstructive Apnea', 'SpO2 artifact','Central Apnea']
            for feature in binary_features:
                if feature in patient_data.columns:
                    patient_data = patient_data.drop(feature, axis=1)
            df1 = patient_data.pop('staging')
            patient_data['staging'] = df1
            n_features = patient_data.shape[1] - 1
            data_columns, hidden_sequence, observation_sequence, train1, test = preprocess_data(data=patient_data)

            test_observation_sequence = train1.iloc[:, 0:n_features].values.tolist()

        predicted_y = classifier.predict(test_observation_sequence)

        conf_hmm = metrics.confusion_matrix(hidden_sequence, predicted_y, states)
        score = accuracy_score(hidden_sequence, predicted_y)
        pt = classification_report(hidden_sequence, predicted_y)
        confs = confs.__add__(conf_hmm)

        #count first predicted wake
        grouped_states = pd.Series([list(group) for key, group in groupby(predicted_y)])
        state_length = [len(grouped_states[i]) for i in range(0, len(grouped_states))]
        if grouped_states[0][0] == 'Wake':

            wake_lengths.append(state_length[0])
        else:
            print('Pacient ' + str(list_of_patients['file_name'][i]) + ' nezacina Wake.')
            wake_lengths.append(0)

        precision_tables.write(pt)
        pts.append(pt)
        print(pt)
        print(conf_hmm)
        cms.append(conf_hmm)
        scores.append(score)
        confusion_matrices.write(str(conf_hmm) + '\n')
        scores_file.write(str(score) + '\n')

print(confs)
precision_tables.close()
scores_file.close()
confusion_matrices.close()

wakes=pd.DataFrame(wake_lengths)
wakes.to_csv('/disk/bios/blaskova/kod_sleep_org/RF_balanced/rf_wlengths' + str(do) + '.csv')


