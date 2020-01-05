import pandas as pd
import numpy as np

def find_supp(fromState, toState, dir):

    path = dir + '/' + fromState + toState + '.csv'
    df = pd.read_csv(path)
    df = df.drop(df.columns[0], axis=1)
    df = df.dropna(how='all', axis=0)

    supp = [df.index[0], df.index[-1]]

    return supp

def load_coeff(path):

    df = pd.read_csv(path, sep=';')

    for row in range(df.shape[0]):
        df['Coeff'][row] = df['Coeff'][row][7:-2]

    df['Coeff'] = df['Coeff'].apply(eval)
    return df

def calc_norm(df, dom):

    f = np.poly1d([0,0,0])
    for row in range(df.shape[0]):
        poly = np.poly1d(df['Coeff'][row])
        f = f + poly

    x = np.arange(dom)
    norm = 1/f(x)

    return norm

states = ['Wake', 'NonREM1', 'NonREM2', 'NonREM3', 'REM'] # prepare states to generate combinations
list_of_pairs = [(p1, p2) for p1 in states for p2 in states] #combinations


########################################################################################################################
# IN CASE OF DIFFERENT SUPPORTS
supports=[]

for transition in list_of_pairs:

    supp = find_supp(transition[0], transition[1],
                     '/Users/kristina/PycharmProjects/vyskumak/diplomka/patients_without_1st_wake/transition_prob_dfs')
    supports.append(supp)

left = [supp[0] for supp in supports]
right = [supp[1] for supp in supports]
########################################################################################################################


