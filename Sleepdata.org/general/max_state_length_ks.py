import pandas as pd
import os
from itertools import groupby
import data_preprocessing_ks as dp

place = 'local'

if place == 'local':
    local_list = "/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/list_of_patients.csv"

    local_pdirectory = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/shhs1'
    forbidden_ps1 = pd.read_csv(
        '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/forbidden_pacients_length_of_wake.csv')

else:
    local_list = "/disk/bios/blaskova/kod_sleep_org/list_of_patients.csv"
    local_pdirectory = '/disk/bios/sleepdata.org/shhs/polysomnography/csv/shhs1'

list_of_patients = pd.read_csv(local_list)
forbidden_ps1['file_name'] = forbidden_ps1['Pacient'] + '.csv'
list_of_patients = list_of_patients[~list_of_patients['file_name'].isin(forbidden_ps1['file_name'])].reset_index(drop=True)

stavy = []
dlzky = []
pacient = []
celkova_dlzka_spanku = []

for i in range(0,list_of_patients.shape[0]):

    path = str(local_pdirectory) + "/" + str(list_of_patients['file_name'][i])

    patient_data = dp.data_import(path,['staging'])
    hidden_sequence = patient_data['staging'].values.tolist()
    del patient_data

    grouped_states = pd.Series([list(group) for key, group in groupby(hidden_sequence)])
    dlzka_stavu = [len(grouped_states[i]) for i in range(0, len(grouped_states))]
    stav = [grouped_states[i][0] for i in range(0, len(grouped_states))]

    stavy.extend(stav)
    dlzky.extend(dlzka_stavu)
    pacient.extend([path[7:-4] for i in range(0, len(stav))])
    celkova_dlzka_spanku.extend([len(hidden_sequence) for i in range(0, len(stav))])

df = pd.DataFrame({'state': stavy, 'State length': dlzky, 'Pacient': pacient, 'Sleep length': celkova_dlzka_spanku})
#df.to_csv('/disk/bios/blaskova/kod_sleep_org/results/stage_stats.csv',index=False)


weights = np.ones_like(wake['Minutes']) / float(len(wake['Minutes']) / 100)
plt.hist(wake['Minutes'], bins=np.arange(min(wake['Minutes']), max(wake['Minutes']) + binwidth, binwidth),
         alpha=0.5, label='Wake', color='r', weights=weights,histtype='step', stacked=True, fill=False,linewidth=5.0)

weights = np.ones_like(nonrem1['Minutes']) / float(len(nonrem1['Minutes']) / 100)
plt.hist(nonrem1['Minutes'], bins=np.arange(min(nonrem1['Minutes']), max(nonrem1['Minutes']) + binwidth, binwidth),
         alpha=0.5, label='NonREM1', color='b', weights=weights,histtype='step', stacked=True, fill=False,linewidth=5.0)

weights = np.ones_like(nonrem2['Minutes']) / float(len(nonrem2['Minutes']) / 100)
plt.hist(nonrem2['Minutes'], bins=np.arange(min(nonrem2['Minutes']), max(nonrem2['Minutes']) + binwidth, binwidth),
         alpha=0.5, label='NonREM2', color='g', weights=weights, histtype='step', stacked=True, fill=False,linewidth=5.0)

weights = np.ones_like(nonrem3['Minutes']) / float(len(nonrem3['Minutes']) / 100)
plt.hist(nonrem3['Minutes'], bins=np.arange(min(nonrem3['Minutes']), max(nonrem3['Minutes']) + binwidth, binwidth),
         alpha=0.5, label='NonREM3', color='yellow', weights=weights,histtype='step', stacked=True, fill=False,linewidth=5.0)
weights = np.ones_like(rem['Minutes']) / float(len(rem['Minutes']) / 100)

plt.hist(rem['Minutes'], bins=np.arange(min(rem['Minutes']), max(rem['Minutes']) + binwidth, binwidth),
         alpha=0.5, label='REM', color='brown', weights=weights,histtype='step', stacked=True, fill=False,linewidth=5.0)
