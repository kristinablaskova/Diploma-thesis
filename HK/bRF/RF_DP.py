import pandas as pd
import sklearn.model_selection as ms
import os
import data_preprocessing as dp
from sklearn.ensemble import RandomForestClassifier
import numpy as np
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
    precision_tables = open('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF_balanced/results/pt.txt', 'w')
    confusion_matrices = open('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF_balanced/results/cms.txt', 'w')
    scores_file = open('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF_balanced/results/scores.txt', 'w')
    pred_dir='/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF_balanced/results/predicted_states/'
    forbidden_ps1 = pd.read_csv(
        '/Users/kristina/PycharmProjects/vyskumak/diplomka/forbidden_pacients_length_of_wake.csv')

else:
    data_path = server_data
    plist = server_list
    precision_tables = open('/disk/bios/blaskova/vysledky/rf_pt.txt', 'w')
    confusion_matrices = open('/disk/bios/blaskova/vysledky/rf_cms.txt', 'w')
    scores_file = open('/disk/bios/blaskova/vysledky/rf_scores.txt', 'w')

states = ['NonREM1', "NonREM2", "NonREM3", "REM", "Wake"]
confs = np.zeros((len(states), len(states)))
cms=[]
pts=[]
scores=[]


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


    return train_df, training_class_array, feature_names,

########################################################################################################################


#LOAD PATIENT LIST AND SPLIT IT TO TRAINING AND TESTING SET
list_of_patients = pd.read_csv(plist)
forbidden_ps1['file_name'] = forbidden_ps1['Pacient'] + '.csv'
list_of_patients = list_of_patients[~list_of_patients['file_name'].isin(forbidden_ps1['file_name'])].reset_index(drop=True)

print("List of patients loaded")

for i in range(0, list_of_patients.shape[0]):
    print("Predicting the patient number " + str(i))

    # * prepare traning and testing datasets *

    list_of_testing_patients = list_of_patients.iloc[[i]]
    list_of_training_patients = list_of_patients.drop(list_of_patients.index[i], axis=0)

    list_of_training_patients = list_of_training_patients.reset_index()

    train_df, training_class_array, feature_names = prepare_training(list_of_training_patients)

    y = train_df['hypnogram_User'].values.tolist()
    X = train_df.values[:,:-1]

    classifier = RandomForestClassifier(n_estimators=10, max_depth=10,class_weight='balanced')
    classifier.fit(X, y)

    list_of_testing_patients = list_of_testing_patients.reset_index()

    for k in range(list_of_testing_patients.shape[0]):
        print(list_of_testing_patients['file_name'])
        print(str(list_of_testing_patients['file_name'][k]))
        print(data_path)
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

        predicted_y = classifier.predict(test_observation_sequence)

        conf_hmm = metrics.confusion_matrix(hidden_sequence, predicted_y, states)
        score = accuracy_score(hidden_sequence, predicted_y)
        pt = classification_report(hidden_sequence, predicted_y)
        confs = confs.__add__(conf_hmm)

        precision_tables.write(pt)
        pts.append(pt)
        print(pt)
        print(conf_hmm)
        cms.append(conf_hmm)
        scores.append(score)
        confusion_matrices.write(str(conf_hmm) + '\n')
        scores_file.write(str(score) + '\n')
        pred = pd.DataFrame(predicted_y)
        pred.to_csv(pred_dir + str(list_of_testing_patients['file_name'][k]),index=False)

print(confs)
precision_tables.close()
scores_file.close()
confusion_matrices.close()


