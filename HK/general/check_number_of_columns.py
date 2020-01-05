import pandas as pd
import os
import data_preprocessing as dp

list_of_patients = pd.read_csv("/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv")
directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')
path = str(directory)[2:-1] + "/" + str(list_of_patients['file_name'][0])
patient_data = dp.data_import(path)
list_of_columns = patient_data.columns.values.tolist()

for i in range(list_of_patients.shape[0]):
    path = str(directory)[2:-1] + "/" + str(list_of_patients['file_name'][i])
    patient_data = dp.data_import(path)
    if patient_data.shape[1] != 48:
        wrong_list_columns = patient_data.columns.values.tolist()
        print(str(list_of_patients['file_name'][i]))
        b3 = [val for val in list_of_columns if val not in wrong_list_columns]
        print(b3)
