import os
import pandas as pd
import hmm as myhmm
import matplotlib.pyplot as plt
import data_preprocessing as dp

#hypnogram of average person
average_person = dp.data_import('/Users/kristina/PycharmProjects/vyskumak/Data/12.10.2016-Z-M-39let.csv')
average_person['time'] = [(0.5/60*i) for i in range(0, 973)]
cleanup_nums = {"hypnogram_User": {"Wake": 5, "REM": 4, "NonREM1": 3, "NonREM2": 2, "NonREM3": 1}}
average_person.replace(cleanup_nums, inplace=True)
plt.plot(average_person['time'], average_person['hypnogram_User'])
y = [5, 4, 3, 2, 1]
labels = ["Wake", "REM", "NonREM1", "NonREM2", "NonREM3"]
plt.yticks(y, labels)
plt.xlabel('Measurement length [hours]')
plt.ylabel('Sleep stage')
plt.title('Hypnogram of average person from dataset Hradec Kralove')
plt.show()


#age histogram
plt.hist(males['age'],10, color='grey')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of age from Hradec Kralove dataset')
plt.axvline(list_of_patients_with_attributes['age'].mean(), color='orange', linestyle='dashed', linewidth=1)
plt.axvline(list_of_patients_with_attributes['age'].median(), color='yellow', linestyle='dashed', linewidth=1)
plt.show()

#sex histogram
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.title('Histogram of sex from Hradec Kralove dataset')
plt.xticks(rotation=90)
list_of_patients_with_attributes['sex'].value_counts().plot(kind = 'bar', color='grey')