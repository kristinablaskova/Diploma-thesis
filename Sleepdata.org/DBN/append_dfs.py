import pandas as pd

dir='/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/prob_dfs/no_wake_no_outliers/'
states = ['Wake', 'NonREM1', 'NonREM2', 'NonREM3', 'REM'] # prepare states to generate combinations
list_of_pairs = [(p1, p2) for p1 in states for p2 in states] #combinations


def load_tdf(pair):
    df = pd.DataFrame()
    for i in range(1, 12):
        df1 = pd.read_csv(dir + str(list_of_pairs[pair][0] + list_of_pairs[pair][1]) + str(i) + '.csv')
        df = df.append(df1)
        df.reset_index(drop=True,inplace=True)

    return df

def transition_dfs():

    dataframes = []

    for pair in range(0, len(list_of_pairs)):
        try:
            vars()[str(list_of_pairs[pair][0] + list_of_pairs[pair][1])] = load_tdf(pair)
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
