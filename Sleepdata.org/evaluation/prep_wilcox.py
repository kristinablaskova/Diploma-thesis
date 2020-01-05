import pandas as pd
import os
import data_preprocessing_ks as dp



directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/RF/paralel_RF/predicted_states')
hs_dir = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/shhs1/'
preds=[]
hs = []

for file in os.listdir(directory):

    try:
        filename = os.fsdecode(file)

        if filename.endswith(".csv"):

            path = str(directory)[2:-1] + "/" + str(filename)
            y_pred = pd.read_csv(path,delim_whitespace=True,header=0)
            preds.extend(y_pred['0'])

            path1 = hs_dir + str(filename)
            y = dp.data_import(path1,['staging'])
            for i in reversed(range(0, len(y['staging']))):
                if y['staging'][i] == "NotScored":
                    y = y.drop([i])
            hs.extend(y['staging'])

            if len(y_pred['0']) != len(y['staging']):
                print(str(filename) + ' ' + str(len(y_pred['0'])) + ' ' + str(len(y['staging'])))



    except:
        pass


difs = [0 if preds[i] != hs[i] else 1 for i in range(0,len(hs))]
dif=pd.DataFrame({'difs':difs})
dif.to_csv(str(directory)[2:-1] + 'dif.csv',index=False)
