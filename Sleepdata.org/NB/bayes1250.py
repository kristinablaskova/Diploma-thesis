import pandas as pd
import numpy as np
import dist_ks as dst
import pomegranate as pg
import os
import data_preprocessing_ks as dp
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.metrics import classification_report,accuracy_score

od=1200
do=1250

place = 'server'

local_data = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/local_test_set/'
local_list = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/list_of_patients.csv'

server_data = "/disk/bios/sleepdata.org/shhs/polysomnography/csv/shhs1/"
server_list = "/disk/bios/blaskova/kod_sleep_org/list_of_patients.csv"

if place == 'local':
    data_path = local_data
    plist = local_list
    precision_tables = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/bayes/results/bayes_pt.txt', 'w')
    confusion_matrices = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/bayes/results/bayes_cms.txt', 'w')
    scores_file = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/bayes/results/bayes_scores.txt', 'w')
    pred_dir = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/bayes/results/predicted_states/'
    cfeatures = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/common_features.txt'
    forbidden_ps1 = pd.read_csv(
        '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/forbidden_pacients_length_of_wake.csv')


else:
    data_path = server_data
    plist = server_list
    pred_dir = '/disk/bios/blaskova/kod_sleep_org/bayes/results/predicted_states/'
    cfeatures = '/disk/bios/blaskova/kod_sleep_org/results/common_features.txt'
    forbidden_ps1 = pd.read_csv(
        '/disk/bios/blaskova/kod_sleep_org/results/forbidden_pacients_length_of_wake.csv')

    precision_tables = open('/disk/bios/blaskova/kod_sleep_org/bayes/results/pt' + str(do) + '.txt', 'w')
    confusion_matrices = open('/disk/bios/blaskova/kod_sleep_org/bayes/results/cms' + str(do) + '.txt', 'w')
    scores_file = open('/disk/bios/blaskova/kod_sleep_org/bayes/results/scores' + str(do) + '.txt', 'w')

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

def prepare_training(list_of_training_patients):
    # creates a giant hidden sequence array and train_df from all training patients

    # create array of hidden state sequence of traning dataset
    training_class_array = []

    # create df of observation sequence of traning dataset
    train_df = pd.DataFrame()

    for j in range(list_of_training_patients.shape[0]):


        path = data_path + str(list_of_training_patients['file_name'][j])

        if os.path.isfile(path) == True:



            data = dp.data_import(path,common_features)
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


#LOAD PATIENT LIST AND SPLIT IT TO TRAINING AND TESTING SET
list_of_patients = pd.read_csv(plist)
forbidden_ps1['file_name'] = forbidden_ps1['Pacient'] + '.csv'
list_of_patients = list_of_patients[~list_of_patients['file_name'].isin(forbidden_ps1['file_name'])].reset_index(drop=True)
print("List of patients loaded")


#len(list_of_patients['file_name'])
for i in range(od, do):
    print("Predicting the patient number " + str(i))

    # * prepare traning and testing datasets *

    list_of_testing_patients = list_of_patients.iloc[[i]]
    list_of_training_patients = list_of_patients.drop(list_of_patients.index[i], axis=0)

    list_of_training_patients = list_of_training_patients.reset_index()

    train_df, training_class_array, feature_names = prepare_training(list_of_training_patients)

    print('Training dataset prepared')


    # * generate observation probabilities *

    hmm_dist = dst.Distributions(train_df)
    dist, state_names = hmm_dist.gauss_kernel_dist(feature_names)
    print('Observation probabilities prepared')

    # * initiate HMM *

    model = pg.NaiveBayes(dist)

    # * test the model

    list_of_testing_patients = list_of_testing_patients.reset_index()

    for k in range(list_of_testing_patients.shape[0]):
        path = data_path + str(list_of_testing_patients['file_name'][k])

        if os.path.isfile(path) == True:

            patient_data = dp.data_import(path,common_features)
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

        path = model.predict(test_observation_sequence)

        conf_hmm = metrics.confusion_matrix(hidden_sequence, [state_names[id] for id in path], states)
        score= accuracy_score(hidden_sequence, [state_names[id] for id in path])
        pt = classification_report(hidden_sequence, [state_names[id] for id in path])
        confs = confs.__add__(conf_hmm)

        precision_tables.write(pt)
        pts.append(pt)
        print(pt)
        print(conf_hmm)
        cms.append(conf_hmm)
        scores.append(score)
        confusion_matrices.write(str(conf_hmm)+'\n')
        scores_file.write(str(score)+ '\n')
        pred = pd.DataFrame([state_names[id] for id in path])
        pred.to_csv(pred_dir + str(list_of_testing_patients['file_name'][k]), index=False)


print(confs)
precision_tables.close()
scores_file.close()
confusion_matrices.close()