import pandas as pd
from itertools import groupby
import data_preprocessing_ks as dp

place='server'

if place == 'local':
    local_list = "/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/list_of_patients.csv"
    local_pdirectory = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/shhs1'
    save_path='/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/no_1st_wake/'




else:
    local_pdirectory = '/disk/bios/sleepdata.org/shhs/polysomnography/csv/shhs1'
    local_list = "/disk/bios/blaskova/kod_sleep_org/list_of_patients.csv"
    save_path = "/disk/bios/blaskova/kod_sleep_org/data_no_wake/"

    common_features=pd.read_csv('/disk/bios/blaskova/kod_sleep_org/results/common_features.txt')
    common_features = [common_features.columns[i].replace(" '", '').replace("['", '').replace("']", '').replace("'", '')
                       for i in range(len(common_features.columns))]

list_of_patients = pd.read_csv(local_list)


# variable containing how long the first wakes are
wake_lengths = []

for i in range(0,list_of_patients.shape[0]):
    print(i)

    path = str(local_pdirectory) + "/" + str(list_of_patients['file_name'][i])

    patient_data = dp.data_import(path,common_features)
    hidden_sequence = patient_data['staging'].values.tolist()

    grouped_states = pd.Series([list(group) for key, group in groupby(hidden_sequence)])
    state_length = [len(grouped_states[i]) for i in range(0, len(grouped_states))]

    if grouped_states[0][0] == 'Wake':

        patient_data.loc[state_length[0]:].to_csv(
            save_path
            + str(list_of_patients['file_name'][i]), index=False, sep=',')
        wake_lengths.append(state_length[0])
    else:
        print('Pacient ' + str(list_of_patients['file_name'][i]) + ' nezacina Wake.')
        patient_data.to_csv(
            save_path
            + str(list_of_patients['file_name'][i]), index=False,sep=',')
        wake_lengths.append(0)
