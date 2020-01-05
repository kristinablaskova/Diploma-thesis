# 1. naucit sa pravdepodobnosti na trenovacich time sliced patients
# 2. fitnut pravdepodobnostne casovo zavisle matice
# 3. vygenerovat transition matrix v case t
# 4. vygenerovat hmm v case t a predikovat na t+1

import pandas as pd
import numpy as np
import dist as dst
import pomegranate as pg
import os
import data_preprocessing as dp
import sklearn.model_selection as ms
from viterbi import viterbi
import sklearn.metrics as metrics
from sklearn.metrics import classification_report,accuracy_score



place = 'local'

local_data = '/Users/kristina/PycharmProjects/vyskumak/Data/'
local_list = '/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv'

server_data = "/disk/bios/blaskova/kod/data/patients/"
server_list = "/disk/bios/blaskova/kod/list_of_patients_with_attributes.csv"

if place == 'local':
    data_path = local_data
    plist = local_list

    forbidden_ps1 = pd.read_csv(
        '/Users/kristina/PycharmProjects/vyskumak/diplomka/forbidden_pacients_length_of_wake.csv')
    pred_dir='/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/orig_hmm/results/predicted_states/'

    precision_tables = open(
        '/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/orig_hmm/results/pt.txt', 'w')
    confusion_matrices = open(
        '/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/orig_hmm/results/cms.txt', 'w')
    maximal_indices = open(
        '/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/orig_hmm/results/HMM_find_a_max_ids.txt',
        'w')
    scores_file = open(
        '/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/orig_hmm/results/scores.txt', 'w')

else:
    data_path = server_data
    plist = server_list
########################################################################################################################
# LEARN TIME DEPENDENT TRANSITION PROBABILITIES
########################################################################################################################
states = ['NonREM1', "NonREM2", "NonREM3", "REM", "Wake"]

confs = np.zeros((len(states), len(states)))
cms=[]
pts=[]
scores=[]
max_ids=[]


########################################################################################################################
# INITIATE HMM
########################################################################################################################


########################################################################################################################
#FUNKCIE
########################################################################################################################

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


        path = data_path + str(list_of_training_patients['file_name'][j])

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

########################################################################################################################

#LOAD PATIENT LIST AND SPLIT IT TO TRAINING AND TESTING SET
list_of_patients = pd.read_csv(plist)
forbidden_ps1['file_name'] = forbidden_ps1['Pacient'] + '.csv'
list_of_patients = list_of_patients[~list_of_patients['file_name'].isin(forbidden_ps1['file_name'])].reset_index(drop=True)

print("List of patients loaded")

for i in range(0, len(list_of_patients['file_name'])):
    print("Predicting the patient number " + str(i))

    # * prepare traning and testing datasets *

    list_of_testing_patients = list_of_patients.iloc[[i]]
    list_of_training_patients = list_of_patients.drop(list_of_patients.index[i], axis=0)

    list_of_training_patients = list_of_training_patients.reset_index()

    train_df, training_class_array, feature_names = prepare_training(list_of_training_patients)

    # learn starting probs
    start_freq = pd.Series([training_class_array[i][0] for i in range(len(training_class_array))])
    start_freq = start_freq.value_counts(normalize=True)
    pi = []
    for state in states:
        try:
            pi.append(start_freq[state])
        except:
            pi.append(0)

    # learn transition probs in case of constant tm

    tm = pd.crosstab(pd.Series(training_class_array[0][:-1], name='From'),
                     pd.Series(training_class_array[0][1:], name='To'))
    for i in range(1, len(training_class_array)):
        tmi = pd.crosstab(pd.Series(training_class_array[i][:-1], name='From'),
                          pd.Series(training_class_array[i][1:], name='To'))
        tm = tm.add(tmi, fill_value=0)

    tm = tm.div(tm.sum(axis=1), axis=0)

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
            patient_data = dp.data_import(path)
            binary_features = ["Gain", "Bradycardia", "LegMovement", "CentralApnea", "Arousal", "Hypopnea",
                                       "RelativeDesaturation", "Snore", "ObstructiveApnea", "MixedApnea", "LongRR", "Tachycardia"]
            for feature in binary_features:
                if feature in patient_data.columns:
                    patient_data = patient_data.drop(feature, axis=1)
            df1 = patient_data.pop('hypnogram_User')
            patient_data['hypnogram_User'] = df1
            n_features = patient_data.shape[1] - 2
            data_columns, hidden_sequence, observation_sequence, train1, test = preprocess_data(data=patient_data)

            test_observation_sequence = train1.iloc[:, 0:n_features].values.tolist()

        emission_prob_array = model.predict_proba(test_observation_sequence)

        #tu klasicky hmm
        path, delta, phi = viterbi(pi, tm, emission_prob_array)
        best_path = path

        conf_hmm = metrics.confusion_matrix(hidden_sequence, [state_names[id] for id in best_path], states)
        score= accuracy_score(hidden_sequence, [state_names[id] for id in best_path])
        pt = classification_report(hidden_sequence, [state_names[id] for id in best_path])
        precision_tables.write(pt)
        pts.append(pt)
        print(pt)
        print(conf_hmm)
        cms.append(conf_hmm)
        scores.append(score)
        confusion_matrices.write(str(conf_hmm)+'\n')
        scores_file.write(str(score)+ '\n')
        pred = pd.DataFrame([state_names[id] for id in best_path])
        pred.to_csv(pred_dir + str(list_of_testing_patients['file_name'][k]),index=False)

        confs = confs.__add__(conf_hmm)

#confs = confs.astype('float') / confs.sum(axis=1)[:, np.newaxis]
print(confs)
precision_tables.close()
scores_file.close()
maximal_indices.close()
confusion_matrices.close()