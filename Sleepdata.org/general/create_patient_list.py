import pandas as pd
import os

directory = os.fsencode('/disk/bios/sleepdata.org/shhs/polysomnography/csv/shhs1')
list_of_patients = pd.DataFrame(columns=['file_name'])

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        list_of_patients = list_of_patients.append({'file_name' : str(filename)}, ignore_index=True )

list_of_patients.to_csv('/disk/bios/blaskova/kod_sleep_org/list_of_patients.csv', index=False)