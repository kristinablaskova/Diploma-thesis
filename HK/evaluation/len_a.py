import pandas as pd
import os
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt


data_path = '/Users/kristina/PycharmProjects/vyskumak/Data/'
forbidden_ps1 = pd.read_csv(
    '/Users/kristina/PycharmProjects/vyskumak/diplomka/forbidden_pacients_length_of_wake.csv')
local_list = '/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv'
plist = local_list

#LOAD PATIENT LIST AND DROP OUTLIERS
list_of_patients = pd.read_csv(plist)
forbidden_ps1['file_name'] = forbidden_ps1['Pacient'] + '.csv'
list_of_patients = list_of_patients[~list_of_patients['file_name'].isin(forbidden_ps1['file_name'])].reset_index(drop=True)

# variable containing how long the first wakes are
wake_lengths = []

for i in range(0,list_of_patients.shape[0]):

    path = data_path + str(list_of_patients['file_name'][i])

    if os.path.isfile(path) == True:

        patient_data = pd.read_csv(path,sep=';')
        patient_data = patient_data.replace(',', '.', regex=True)
        patient_data.columns = [c.replace('.', '_') for c in patient_data.columns]
        patient_data = patient_data.loc[:, (patient_data != 0).any(axis=0)]
        hidden_sequence = patient_data['hypnogram_User'].values.tolist()
        del patient_data

        grouped_states = pd.Series([list(group) for key, group in groupby(hidden_sequence)])
        state_length = [len(grouped_states[i]) for i in range(0, len(grouped_states))]

        if grouped_states[0][0] == 'Wake':

            # patient_data.loc[:state_length[0]].to_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/patients_wake_only/'
            #                                            + str(filename)[:-4] + '.csv', index=False, sep = ';')
            wake_lengths.append(state_length[0])
        else:
            print('Pacient nezacina Wake.')
            wake_lengths.append(0)

real = pd.DataFrame(wake_lengths)

predicted = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/modified_hmm/results/HMM_find_a_max_ids.txt',
                        delim_whitespace=True,header=None)

def plot_hist(real,prediced,binwidth):

    weights = np.ones_like(real / 2.0) / float(len(real / 2.0) / 100.0)
    plt.hist(real / 2.0, bins=np.arange(min(real / 2.0), max(real / 2.0) + binwidth, binwidth),
             alpha=0.5, label='Real', color='g', weights=weights)

    weights = np.ones_like(prediced[0] / 2.0) / float(len(prediced[0] / 2.0) / 100.0)
    plt.hist(prediced[0] / 2.0, bins=np.arange(min(prediced[0] / 2.0), max(prediced[0] / 2.0) + binwidth, binwidth),
             alpha=0.5, label='Predicted', color='b', weights=weights)

    plt.xlabel('Initial Wake duration [min.]')
    plt.ylabel('Fraction of patients [%]')
    plt.legend()
