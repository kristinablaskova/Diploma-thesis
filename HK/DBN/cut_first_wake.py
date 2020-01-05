import pandas as pd
import os
from itertools import groupby


directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')
forbidden_ps1 = pd.read_csv(
    '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/forbidden_pacients_length_of_wake.csv')

# variable containing how long the first wakes are
wake_lengths = []

for file in os.listdir(directory):

    filename = os.fsdecode(file)
    if filename.endswith(".csv") and forbidden_ps1[forbidden_ps1['Pacient'].str.contains(filename[:-4])].empty == True:

        path =str(directory)[2:-1]+"/"+str(filename)

        patient_data = pd.read_csv(path, sep=";")
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
            print('Pacient ' + str(filename) + ' nezacina Wake.')

