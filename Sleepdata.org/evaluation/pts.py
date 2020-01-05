import pandas as pd

def read_pts(dir):
    data = pd.DataFrame()
    for do in [50,350,650,950,1250]:

        path = dir + str(do) + '.txt'
        df = pd.read_csv(path, header=0,
                           delim_whitespace = True)
        data = data.append(df)

    return data

dir = '/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/paralel_timeHMM/pt'
data = read_pts(dir)

data = data[data.index != 'precision']
data = data.apply(pd.to_numeric)

mean_precision = data[data.index == 'avg/total']["precision"].mean() * 100
mean_recall = data[data.index == 'avg/total']["recall"].mean() * 100
mean_f1 = data[data.index == 'avg/total']["f1-score"].mean() * 100

std_precision = data[data.index == 'avg/total']["precision"].std() * 100
std_recall = data[data.index == 'avg/total']["recall"].std() * 100
std_f1 = data[data.index == 'avg/total']["f1-score"].std() * 100

nonrem1 = data[data.index == 'NonREM1']
nonrem2 = data[data.index == 'NonREM2']
nonrem3 = data[data.index == 'NonREM3']
wake = data[data.index == 'Wake']
rem = data[data.index == 'REM']
