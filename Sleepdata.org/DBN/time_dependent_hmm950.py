# 1. naucit sa pravdepodobnosti na trenovacich time sliced patients
# 2. fitnut pravdepodobnostne casovo zavisle matice
# 3. vygenerovat transition matrix v case t
# 4. vygenerovat hmm v case t a predikovat na t+1

import generate_tm_in_time as gtm
import pandas as pd
import numpy as np
import dist_ks as dst
import pomegranate as pg
import os
import data_preprocessing_ks as dp
import sklearn.model_selection as ms
from modified_viterbi import viterbi_t
from viterbi import viterbi
import sklearn.metrics as metrics
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
from itertools import groupby

od=900
do=950

place = 'server'

local_data = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/local_test_set/'
local_list = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/list_of_patients.csv'
local_coeff = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/prob_dfs/no_wake_no_outliers/coeffs.csv'

server_data = "/disk/bios/sleepdata.org/shhs/polysomnography/csv/shhs1/"
server_list = "/disk/bios/blaskova/kod_sleep_org/list_of_patients.csv"
server_coeff = '/disk/bios/blaskova/kod_sleep_org/prob_dfs/no_wake_no_outliers/coeffs.csv'

local_cfeatures = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/common_features.txt'
server_cfeatures = '/disk/bios/blaskova/kod_sleep_org/results/common_features.txt'

if place == 'local':
    data_path = local_data
    plist = local_list
    coeff_path = local_coeff
    cfeatures = local_cfeatures
    forbidden_ps1 = pd.read_csv(
        '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/forbidden_pacients_length_of_wake.csv')
    pred_dir='/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/time_HMM/results/predicted_states/'

    precision_tables = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/time_HMM/results/pt.txt', 'w')
    confusion_matrices = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/time_HMM/results/cms.txt', 'w')
    maximal_indices = open(
        '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/time_HMM/results/HMM_find_a_max_ids.txt', 'w')
    scores_file = open('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/time_HMM/results/scores.txt', 'w')

else:
    data_path = server_data
    plist = server_list
    coeff_path = server_coeff
    cfeatures = server_cfeatures
    pred_dir='/disk/bios/blaskova/kod_sleep_org/time_HMM/results/predicted_states/'
    forbidden_ps1 = pd.read_csv(
        '/disk/bios/blaskova/kod_sleep_org/results/forbidden_pacients_length_of_wake.csv')

    maximal_indices = open(
        '/disk/bios/blaskova/kod_sleep_org/time_HMM/results/HMM_find_a_max_ids'+ str(do) +'.txt', 'w')
    precision_tables = open('/disk/bios/blaskova/kod_sleep_org/time_HMM/results/pt' + str(do) + '.txt', 'w')
    confusion_matrices = open('/disk/bios/blaskova/kod_sleep_org/time_HMM/results/cms' + str(do) + '.txt', 'w')
    scores_file = open('/disk/bios/blaskova/kod_sleep_org/time_HMM/results/scores' + str(do) + '.txt', 'w')

common_features = pd.read_csv(cfeatures)
common_features=[common_features.columns[i].replace(" '",'').replace("['",'').replace("']",'').replace("'",'') for i in range(len(common_features.columns))]

########################################################################################################################
# LEARN TIME DEPENDENT TRANSITION PROBABILITIES
########################################################################################################################
states = ['NonREM1', "NonREM2", "NonREM3", "REM", "Wake"]
polynomial_coeffs = gtm.load_coeff(coeff_path)

print("precalculating the transition probabilities...")
T = 1100
tms = []
for t in np.arange(0,T,0.5):
    tm = gtm.get_tm(t, states, polynomial_coeffs)
    tms.append(tm)

    if t % 100 == 0:
        print("...")

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
    #
    # tm = pd.crosstab(pd.Series(training_class_array[0][:-1], name='From'),
    #                  pd.Series(training_class_array[0][1:], name='To'))
    # for i in range(1, len(training_class_array)):
    #     tmi = pd.crosstab(pd.Series(training_class_array[i][:-1], name='From'),
    #                       pd.Series(training_class_array[i][1:], name='To'))
    #     tm = tm.add(tmi, fill_value=0)
    #
    # tm = tm.div(tm.sum(axis=1), axis=0)

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

        emission_prob_array = model.predict_proba(test_observation_sequence)
        #tu novy hmm
        path = model.predict(test_observation_sequence)
        best_path_probs=[]
        paths=[]
        for a in range(0,240,1):
            path, delta, phi, max_path_prob = viterbi_t(pi, tms, emission_prob_array, a)
            best_path_probs.append(max_path_prob)
            paths.append(path)
            if a % 10 == 0:
                print(a)
                print(best_path_probs[a])

        id_max = best_path_probs.index(max(best_path_probs))
        best_path = paths[id_max]

        grouped_states = pd.Series([list(group) for key, group in groupby(hidden_sequence)])
        state_length = [len(grouped_states[i]) for i in range(0, len(grouped_states))]

        if grouped_states[0][0] == 'Wake':

            skutocna_dlzka = state_length[0]
        else:
            skutocna_dlzka=0
        plt.plot(best_path_probs,label='Path probability')
        plt.xlabel('Minutes')
        plt.ylabel('Log-Probability')
        plt.title('Path probability with respect to initial Wake duration')
        plt.plot([id_max/2.0 for max in best_path_probs],best_path_probs, color='red', label='Prediction', linestyle='--')
        plt.plot([skutocna_dlzka/2.0 for max in best_path_probs],best_path_probs, color='green', label='Reality', linestyle='--')
        plt.legend()
        plt.savefig( '/disk/bios/blaskova/kod_sleep_org/time_HMM/results/plots_a/' + str(i) + '.png')
        plt.close()
        #tu klasicky hmm
        # path, delta, phi = viterbi(pi, tm, emission_prob_array)
        # best_path = path

        conf_hmm = metrics.confusion_matrix(hidden_sequence, [state_names[id] for id in best_path], states)
        score= accuracy_score(hidden_sequence, [state_names[id] for id in best_path])
        pt = classification_report(hidden_sequence, [state_names[id] for id in best_path])
        precision_tables.write(pt)
        pts.append(pt)
        print(pt)
        print(conf_hmm)
        cms.append(conf_hmm)
        scores.append(score)
        max_ids.append(id_max)
        confusion_matrices.write(str(conf_hmm)+'\n')
        maximal_indices.write(str(id_max) + '\n')
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