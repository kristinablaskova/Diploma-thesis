import pandas as pd
import numpy as np

czech_patients = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/experimenty/list_of_patients_with_attributes.csv')

sleeporg_patients = pd.read_excel('/Users/kristina/PycharmProjects/vyskumak/diplomka/results/SleepDataDescription.xlsx')

filtered_sleeporg_patients = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/results/list_of_patients.csv')

# rename columns of the dataframes so that merges can be made
filtered_sleeporg_patients['nsrrid'] = filtered_sleeporg_patients['file_name'].str[6:-4]
filtered_sleeporg_patients['nsrrid'] = pd.to_numeric(filtered_sleeporg_patients['nsrrid'])

czech_patients['gender'] = czech_patients['sex']
dict = {'M' :0, 'F': 1}
czech_patients = czech_patients.replace({'gender': dict})
czech_patients['age_s1'] = czech_patients['age']

# add gender and sex attributes to the filtered sleeporg patients dataframe
filtered_sleeporg_patients = pd.merge(filtered_sleeporg_patients, sleeporg_patients, how='inner', on='nsrrid')

df= pd.DataFrame(columns=['czech_p', 'sleeporg_p', 'age', 'gender'])

for idx, patient in czech_patients.iterrows():


    selected_men = filtered_sleeporg_patients[
        np.logical_and(filtered_sleeporg_patients['age_s1'] == patient['age_s1'], filtered_sleeporg_patients['gender'] == patient['gender'])].reset_index()

    if selected_men.empty == False:

        print(patient)

        selected_man = selected_men.iloc[0]

        print(selected_man)

        df = df.append({
            'czech_p': patient['file_name'],
            'czech_age': patient['age'],
            'sleeporg_p': selected_man['file_name'],
            'age': selected_man['age_s1'],
            'gender': selected_man['gender']
        },
        ignore_index=True)

        filtered_sleeporg_patients = filtered_sleeporg_patients.drop(selected_man['index'])

        czech_patients.drop(idx, inplace=True)

young_filtered_sleeporg_patients = filtered_sleeporg_patients[filtered_sleeporg_patients['age_s1'] < 60]

for idx, patient in czech_patients.iterrows():

    selected_men = young_filtered_sleeporg_patients[young_filtered_sleeporg_patients['gender'] == patient['gender']].reset_index()

    print(patient)

    selected_man = selected_men.sample(n=1).iloc[0]

    print(selected_man)

    df = df.append({
        'czech_p': patient['file_name'],
        'czech_age': patient['age'],
        'sleeporg_p': selected_man['file_name'],
        'age': selected_man['age_s1'],
        'gender': selected_man['gender']
    },
    ignore_index=True)

    young_filtered_sleeporg_patients = young_filtered_sleeporg_patients.drop(selected_man['index'])
