import pandas as pd
from scipy.stats import wilcoxon


balanced_rf = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF_balanced/results/predicted_statesdif.csv',
                 delim_whitespace=True)

hmm= pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/orig_hmm/results/predicted_statesdif.csv',
                 delim_whitespace=True)

bayes = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/bayes/results/predicted_statesdif.csv',
                 delim_whitespace=True)

knn = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/knn/results/predicted_statesdif.csv',
                 delim_whitespace=True)

balanced_knn = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/knn_balanced/results/predicted_statesdif.csv',
                 delim_whitespace=True)

rf = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF/results/predicted_statesdif.csv',
                 delim_whitespace=True)

dbn= pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/modified_hmm/results/predicted_statesdif.csv',
                 delim_whitespace=True)

methods = ['balanced_rf', 'hmm', 'bayes', 'knn', 'balanced_knn', 'rf', 'dbn']

method_pairs = [(p1, p2) for p1 in methods for p2 in methods]

for pair in range(0,len(method_pairs)):
    print(str(method_pairs[pair]))

    vars()[str(method_pairs[pair][0] + method_pairs[pair][1] + 'stats')], vars()[str(method_pairs[pair][0] + method_pairs[pair][1] + 'p')] = \
        wilcoxon(vars()[method_pairs[pair][0]]['difs'], vars()[method_pairs[pair][1]]['difs'])

    del vars()[str(method_pairs[pair][0] + method_pairs[pair][1] + 'stats')]

    print(str(method_pairs[pair]) + ' ' + str(vars()[str(method_pairs[pair][0] + method_pairs[pair][1] + 'p')]))
