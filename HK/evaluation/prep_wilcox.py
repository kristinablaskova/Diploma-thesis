import pandas as pd
import os
import data_preprocessing as dp



directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF_balanced/results/predicted_states')
hs_dir = '/Users/kristina/PycharmProjects/vyskumak/Data/'
preds=[]
hs = []
forbidden_ps1 = pd.read_csv(
    '/Users/kristina/PycharmProjects/vyskumak/diplomka/forbidden_pacients_length_of_wake.csv')

for file in os.listdir(directory):

    try:
        filename = os.fsdecode(file)

        if filename.endswith(".csv") and forbidden_ps1[forbidden_ps1['Pacient'].str.contains(filename[:-4])].empty == True:

            path = str(directory)[2:-1] + "/" + str(filename)
            y_pred = pd.read_csv(path,delim_whitespace=True,header=0)
            preds.extend(y_pred['0'])

            path1 = hs_dir + str(filename)
            y = dp.data_import(path1)
            for i in reversed(range(0, len(y['hypnogram_User']))):
                if y['hypnogram_User'][i] == "NotScored":
                    y = y.drop([i])
            hs.extend(y['hypnogram_User'])

            if len(y_pred['0']) != len(y['hypnogram_User']):
                print(str(filename) + ' ' + str(len(y_pred['0'])) + ' ' + str(len(y['hypnogram_User'])))



    except:
        pass


difs = [0 if preds[i] != hs[i] else 1 for i in range(0,len(hs))]
dif=pd.DataFrame({'difs':difs})
dif.to_csv(str(directory)[2:-1] + 'dif.csv',index=False)
