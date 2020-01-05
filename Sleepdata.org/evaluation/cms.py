import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd

def read_cms(dir):
    list_cms = []
    for do in [50,350,650,950,1250]:

        path = dir + str(do) + '.txt'
        cms = np.loadtxt(path)
        for i in range(0, 250, 5):
            list_cms.append(cms[i:i + 5])

    return list_cms

list_cms = read_cms('/Users/kristina/PycharmProjects/vyskumak/kod_sleep_org/RF/paralel_RF/rf_cms')

conf = np.zeros((5,5))
for i in range(0, len(list_cms)):
    conf = conf + list_cms[i]

states = ['NonREM1', "NonREM2", "NonREM3", "REM", "Wake"]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        print("Normalized confusion matrix")
        title='Normalized confusion matrix [%]'
        fmt = '.2f'
    else:
        print('Confusion matrix')
        fmt = '.0f'
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
