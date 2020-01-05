import pandas as pd

knn_balanced = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/knn_balanced/results/scores.txt', header=None,
                   delim_whitespace = True, names=['a'])

modified_hmm = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/modified_hmm/results/scores.txt', header=None,
                   delim_whitespace = True, names=['a'])

hmm = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/orig_hmm/results/scores.txt', header=None,
                   delim_whitespace = True, names=['a'])

rf = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF/results/scores.txt', header=None,
                   delim_whitespace = True, names=['a'])

rf_balanced = pd.read_csv('/Users/kristina/PycharmProjects/vyskumak/diplomka/final_experiments/RF_balanced/results/scores.txt', header=None,
                   delim_whitespace = True, names=['a'])

plt.hist(hmm['a']*100,bins=np.arange(min(hmm['a']*100),
                                          max(hmm['a']*100) + binwidth, binwidth),alpha=0.5,label='HMM')

plt.hist(modified_hmm['a']*100,bins=np.arange(min(modified_hmm['a']*100),
                                          max(modified_hmm['a']*100) + binwidth, binwidth),alpha=0.5,label='Time asleep DBN')

plt.hist(knn_balanced['a']*100,bins=np.arange(min(knn_balanced['a']*100),
                                          max(knn_balanced['a']*100) + binwidth, binwidth),alpha=0.5,label='kNN balanced')

plt.hist(knn['a']*100,bins=np.arange(min(knn['a']*100), max(knn['a']*100) + binwidth, binwidth),alpha=0.5,label='kNN')

plt.hist(bayes['a']*100,bins=np.arange(min(bayes['a']*100), max(bayes['a']*100) + binwidth, binwidth),alpha=0.5,label='Naive Bayes')

