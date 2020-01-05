import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_scores(dir):
    data = pd.DataFrame()
    for do in [50,350,650,950,1250]:

        path = dir + str(do) + '.txt'
        df = pd.read_csv(path, header=None,delim_whitespace=True)
        data = data.append(df,ignore_index=True)

    return data

def plot_hist(df1,df2,binwidth):

    weights = np.ones_like(df2[0]) / float(len(df2[0]) / 100)
    plt.hist(df2[0] * 100, bins=np.arange(min(df2[0] * 100), max(df2[0] * 100) + binwidth, binwidth),
             alpha=0.5, label='Sleepdata.org', color='r',weights=weights)

    weights1 = np.ones_like(df1[0]) / float(len(df1[0])/100)
    plt.hist(df1[0] * 100, bins=np.arange(min(df1[0] * 100), max(df1[0] * 100) + binwidth, binwidth),
             alpha=0.5, label='Hradec Kralove', color='b',weights=weights1)

    ticks = [i for i in range(10,101,10)]

    plt.legend()
    plt.xlabel('Accuracy [%]')
    plt.ylabel('Fraction of patients [%]')
    plt.xticks(ticks)
    plt.xlim(0,100)
    #plt.ylim(0,20)


dir = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/paralel_timeHMM/scores'
data_s = read_scores(dir)
rf = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/modified_hmm/results/HMM_find_a_max_ids.txt', header=None,
                   delim_whitespace = True)