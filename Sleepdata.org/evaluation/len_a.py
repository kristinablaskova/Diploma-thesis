import pandas as pd
import os
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt


data_path = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/shhs1/'
forbidden_ps1 = pd.read_csv(
    '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/forbidden_pacients_length_of_wake.csv')
local_list = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/list_of_patients.csv'
plist = local_list

#LOAD PATIENT LIST AND DROP OUTLIERS
list_of_patients = pd.read_csv(plist)
forbidden_ps1['file_name'] = forbidden_ps1['Pacient'] + '.csv'
list_of_patients = list_of_patients[~list_of_patients['file_name'].isin(forbidden_ps1['file_name'])].reset_index(drop=True)
nums=[i for i in range(0,50)] + [i for i in range(300,350)] + [i for i in range(600,650)] + \
     [i for i in range(900,950)] + [i for i in range(1200,1250)]

# variable containing how long the first wakes are
wake_lengths = []

for i in nums:

    path = data_path + str(list_of_patients['file_name'][i])

    if os.path.isfile(path) == True:

        patient_data = pd.read_csv(path)
        patient_data = patient_data.replace(',', '.', regex=True)
        patient_data.columns = [c.replace('.', '_') for c in patient_data.columns]
        patient_data = patient_data.loc[:, (patient_data != 0).any(axis=0)]
        mapping = {6: "NotScored", 5: "Wake", 4: "REM", 3: "NonREM1", 2: "NonREM2", 1: "NonREM3"}
        patient_data = patient_data.replace({'staging': mapping})
        hidden_sequence = patient_data['staging'].values.tolist()
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



def read_lengths(dir):
    data = pd.DataFrame()
    for do in [50,350,650,950,1250]:

        path = dir + str(do) + '.txt'
        df = pd.read_csv(path, header=None,delim_whitespace=True)
        data = data.append(df,ignore_index=True)

    return data

predicted = read_lengths('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/paralel_timeHMM/HMM_find_a_max_ids')

def plot_hist(real,prediced,binwidth):

    weights = np.ones_like(real / 2.0) / float(len(real / 2.0) / 100.0)
    plt.hist(real / 2.0, bins=np.arange(min(real / 2.0), max(real / 2.0) + binwidth, binwidth),
             alpha=0.5, label='Real', color='g', weights=weights)

    weights = np.ones_like(prediced[0] / 2.0) / float(len(prediced[0] / 2.0) / 100.0)
    plt.hist(prediced[0] / 2.0, bins=np.arange(min(prediced[0] / 2.0), max(prediced[0] / 2.0) + binwidth, binwidth),
             alpha=0.5, label='Predicted', color='r', weights=weights)

    plt.xlabel('Initial Wake duration [min.]')
    plt.ylabel('Fraction of patients [%]')
    plt.legend()
