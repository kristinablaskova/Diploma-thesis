import pandas as pd
import os

directory = os.fsencode('./Data')

df = pd.DataFrame()

for file in os.listdir(directory):

    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        path =str(directory)[2:-1]+"/"+str(filename)

        patient_data = pd.read_csv(path, sep=";")
        patient_data = patient_data.replace(',', '.', regex=True)
        patient_data.columns = [c.replace('.', '_') for c in patient_data.columns]
        patient_data = patient_data.loc[:, (patient_data != 0).any(axis=0)]

        hidden_sequence = patient_data['hypnogram_User'].values.tolist()

        df[path[7:-4]] = pd.Series(hidden_sequence)

df.apply(pd.value_counts)
