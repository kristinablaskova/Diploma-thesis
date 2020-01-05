import pandas as pd

data = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/modified_hmm/results/pt.txt', header=0,
                   delim_whitespace = True)

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
