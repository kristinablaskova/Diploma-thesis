import pandas as pd
import os
from itertools import groupby


directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')
forbidden_ps1 = pd.read_csv(
    '/Users/kristina/PycharmProjects/vyskumak/diplomka/forbidden_pacients_length_of_wake.csv')

stavy = []
dlzky = []
pacient = []
celkova_dlzka_spanku = []

for file in os.listdir(directory):

    filename = os.fsdecode(file)
    if filename.endswith(".csv") and forbidden_ps1[forbidden_ps1['Pacient'].str.contains(filename[:-4])].empty == True:
        path =str(directory)[2:-1]+"/"+str(filename)

        patient_data = pd.read_csv(path, sep=";")
        patient_data = patient_data.replace(',', '.', regex=True)
        patient_data.columns = [c.replace('.', '_') for c in patient_data.columns]
        patient_data = patient_data.loc[:, (patient_data != 0).any(axis=0)]
        hidden_sequence = patient_data['hypnogram_User'].values.tolist()

        grouped_states = pd.Series([list(group) for key, group in groupby(hidden_sequence)])
        dlzka_stavu = [len(grouped_states[i]) for i in range(0, len(grouped_states))]
        stav = [grouped_states[i][0] for i in range(0, len(grouped_states))]

        stavy.extend(stav)
        dlzky.extend(dlzka_stavu)
        pacient.extend([path[7:-4] for i in range(0,len(stav))])
        celkova_dlzka_spanku.extend([len(hidden_sequence) for i in range(0,len(stav))])

df = pd.DataFrame({'state': stavy, 'State length': dlzky, 'Pacient': pacient, 'Sleep length' : celkova_dlzka_spanku})