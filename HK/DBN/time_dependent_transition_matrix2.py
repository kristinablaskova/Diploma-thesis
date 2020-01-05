import pandas as pd
import os

place = 'server'

if place == 'local':

    directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/diplomka/patients_without_1st_wake/data')
    forbidden_ps1 = pd.read_csv(
        '/Users/kristina/PycharmProjects/vyskumak/diplomka/forbidden_pacients_length_of_wake.csv')

    result_dir = '/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/modified_hmm/transition_matrices/prob_dfs/no_outliers/'

else:

    directory = os.fsencode('/disk/bios/blaskova/kod_sleep_org/data_no_wake')
    forbidden_ps1 = pd.read_csv(
        '/disk/bios/blaskova/kod_sleep_org/results/forbidden_pacients_length_of_wake.csv')

    result_dir = '/disk/bios/blaskova/kod_sleep_org/prob_dfs/no_wake_no_outliers/'




states = ['Wake', 'NonREM1', 'NonREM2', 'NonREM3', 'REM'] # prepare states to generate combinations
list_of_pairs = [(p1, p2) for p1 in states for p2 in states] #combinations

safe_size=1100

tms=[]
for i in range(1, safe_size):
    print(i)

    tm = pd.DataFrame()

    for file in os.listdir(directory):
        try:
            filename = os.fsdecode(file)

            if filename.endswith(".csv") and forbidden_ps1[forbidden_ps1['Pacient'].str.contains(filename[:-4])].empty == True:

                path = str(directory)[2:-1] + "/" + str(filename)

                patient_data = pd.read_csv(path, sep=";", skiprows=[j for j in range(1,i)], nrows=60)
                patient_data = patient_data.replace(',', '.', regex=True)
                patient_data.columns = [c.replace('.', '_') for c in patient_data.columns]
                patient_data = patient_data.loc[:, (patient_data != 0).any(axis=0)]

                hidden_sequence = patient_data['hypnogram_User'].values.tolist()
                del patient_data

                tm_patient = pd.crosstab(pd.Series(hidden_sequence[:- 1], name='From'),pd.Series(hidden_sequence[1:], name='To'))

                tm = tm.add(tm_patient, fill_value=0)

        except:

            print('V case i=' + str(i) + 'nasledujuci pacient nespi' + str(filename))

            pass

    tm = tm.div(tm.sum(axis=1), axis=0)
    tms.append(tm)


def extract_transition(tm, pair):
    try:
        proba = tm[str(list_of_pairs[pair][0])][str(list_of_pairs[pair][1])]
    except:
        proba = 'Nan'

    return proba

# generate list of empty dataframes of all possible state combinations
dataframes = []

for pair in range(0, len(list_of_pairs)):
    try:
        vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])] = \
            pd.DataFrame(data={'Prob' : [extract_transition(tms[i],pair) for i in range(0,len(tms))]})

        vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])].to_csv(
            result_dir + str(list_of_pairs[pair][0] + list_of_pairs[pair][1]) + '.csv',index=False)

        dataframes.append(vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])])

    except:

        pass
