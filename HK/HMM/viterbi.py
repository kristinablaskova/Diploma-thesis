import numpy as np
import pandas as pd


# pi = initial probabilities vector, example: pi = [0.6,0.4]
# tms = transition probability matrix, example:
# if          Snow  Rain Sunshine
#   Snow      0.3   0.3      0.4
#   Rain      0.1  0.45     0.45
#   Sunshine  0.2   0.3      0.5
# then a:
# tms = array([[0.3, 0.3, 0.4],
#       [0.1, 0.45, 0.45],
#       [0.2, 0.3, 0.5]], dtype=object)
# emission_prob_array = observation probabilities for states for a given observation at time t:
# emission_prob_array[t] = array([1, 0])
# if snow was observed:
#         Cold  Hot
#Snow        1    0

# viterbi for constant transition matrix
def viterbi(pi, tms, emission_prob_array):
    nStates = np.shape(emission_prob_array)[1]
    T = emission_prob_array.shape[0]
    tms = np.asarray(tms.iloc[:, :].values.tolist())

    # init blank path
    path = np.zeros(T, dtype=int)
    # delta --> aka je maximalna pravdepodobnost, ze sa v case t dostanem do stavu s
    delta = np.zeros((nStates, T))
    # phi --> najpravdepodobnejsi stav v kazdom case t, pre kazdu moznost predchadzajuceho stavu
    phi = np.zeros((nStates, T))

    # initiate initial state distribution
    almost_zero = 0.0001
    pi = np.asarray(pi)  # convert to an array
    pi = np.where(pi == 0., almost_zero, pi)  # if its zero, replace it with 0.0001 = ALMOST ZERO
    pi = pi / np.sum(pi)  # normalize

    # init delta and phi
    delta[:, 0] = np.log(pi) + np.log(emission_prob_array[0])
    phi[:, 0] = np.argmax(pi)

    print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] + np.log(tms[:, s])) + np.log(emission_prob_array[t][s])
            phi[s, t] = np.argmax(delta[:, t - 1] + np.log(tms[:, s]))

    # find optimal path

    path[T - 1] = np.argmax(delta[:, T - 1 ])
    for t in range(T - 2, -1, -1):
        path[t] = phi[path[t +1], [t +1]]

    return path, delta, phi
