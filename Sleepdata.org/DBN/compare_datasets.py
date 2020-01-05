import append_dfs as adf
from plots import plot_proba_pairs
import pandas as pd


states = ['Wake', 'NonREM1', 'NonREM2', 'NonREM3', 'REM'] # prepare states to generate combinations
list_of_pairs = [(p1, p2) for p1 in states for p2 in states] #combinations

def load_old():

    dataframes = []
    dir='/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/modified_hmm/transition_matrices/prob_dfs/no_outliers/'

    for pair in range(0, len(list_of_pairs)):
        try:
            vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])] = \
                pd.read_csv(dir + str(list_of_pairs[pair][0] + list_of_pairs[pair][1]) + '.csv')

            vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])]['Minutes'] = \
                vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])].index/2.0

            vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])] = \
                vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])].replace('Nan', '')

            vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])] = \
                vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])].apply(pd.to_numeric)

            dataframes.append(vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])])

        except:

            pass

    return dataframes

new_dfs = adf.transition_dfs()
old_dfs = load_old()

plot_proba_pairs(new_dfs,old_dfs,list_of_pairs)
