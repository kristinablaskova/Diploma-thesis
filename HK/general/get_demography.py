import os
import pandas as pd

directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/Data')
list_of_patients_with_attributes = pd.DataFrame(['file_name','sex','age'])
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        start = filename.find('-Z-') + 3
        end = filename.find('let', start)
        sex_and_age = filename[start:end]
        sex = sex_and_age[0]
        age = sex_and_age[2:]
        list_of_patients_with_attributes = list_of_patients_with_attributes.append({'file_name': filename,
                                                                                    'sex': sex, 'age': age},
                                                                                   ignore_index=True)

list_of_patients_with_attributes = list_of_patients_with_attributes.dropna(subset=['age'])
list_of_patients_with_attributes = list_of_patients_with_attributes[list_of_patients_with_attributes['age']
                                                                    != '1.2016-AN-M-72']
list_of_patients_with_attributes['age'] = list_of_patients_with_attributes['age'].astype('int')
list_of_patients_with_attributes = list_of_patients_with_attributes.drop(list_of_patients_with_attributes.columns[0], axis=1)

list_of_patients_with_attributes.to_csv(
    '/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv', sep=',')