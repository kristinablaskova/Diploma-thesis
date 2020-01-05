import pandas as pd
import os


directory = os.fsencode('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data')
result = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/data/common_features.txt'

result_txt = open(result,'w')

id=0
for file in os.listdir(directory):

    filename = os.fsdecode(file)

    if filename.endswith(".csv"):
        id = id+1
        path =str(directory)[2:-1]+"/"+str(filename)

        if id==1:

            common_features = pd.read_csv(path,index_col=0,nrows=0)

        else:

            features = pd.read_csv(path,index_col=0,nrows=0)
            common_features = list(set(features) & set(common_features))


result_txt.write(str(common_features))
print(common_features)
result_txt.close()