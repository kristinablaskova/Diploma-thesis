import numpy as np


# viterbi for time dependent transition matrix
def viterbi_t(pi, tms, emission_prob_array, a):

    almost_zero = 0.0001
    nStates = np.shape(emission_prob_array)[1]  # number of states as from the dimensions of emission probability array
    tms = [np.asarray(tm.iloc[:, :].values.tolist()) for tm in
           tms]  # converts list of transition matrices (dataframes)
    # to list of ndarrays

    T = emission_prob_array.shape[0]-a  # the sleep duration
    tau = T + a # the real time

    # init blank path & path probabilities
    path = np.zeros(tau, dtype=int)
    path_prob = np.zeros(tau, dtype=float)
    # delta --> aka je maximalna pravdepodobnost, ze sa v case t dostanem do stavu s
    delta = np.zeros((nStates, T))
    # phi --> najpravdepodobnejsi stav v kazdom case t, pre kazdu moznost predchadzajuceho stavu
    phi = np.zeros((nStates, T))

    # initiate initial state distribution
    pi = np.asarray(pi)  # convert to an array
    pi = np.where(pi == 0., almost_zero, pi)  # if its zero, replace it with 0.0001 = ALMOST ZERO
    pi = pi / np.sum(pi)  # normalize

    # init delta and phi
    delta[:, 0] = np.log(pi) + np.log(emission_prob_array[a])
    phi[:, 0] = np.argmax(pi)

    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] + np.log(tms[t][:, s])) + np.log(emission_prob_array[t+a][s])
            phi[s, t] = np.argmax(delta[:, t - 1] + np.log(tms[t][:, s]))

    # find optimal path

    path[tau - 1] = np.argmax(delta[:, T - 1])
    path_prob[tau - 1] = np.max(delta[:, T - 1])

    # find the best path for times from a to T
    for t in range(tau - 2, -1 + a, -1):
        # mozno zle
        #path[t] = phi[path[t + 1 - a], [t + 1 - a]]
        path[t] = phi[path[t + 1], [t - a + 1]]
        #mozno zle
        # path_prob[t - a] = np.max(delta[:, t - a])

    path[:a] = 4 # assign Wake to the first a steps

    max_path_prob = path_prob[tau - 1] + np.sum(np.log(emission_prob_array[:a,4])) + np.log(tms[0][4, path[a]]) # calculate the path log probability

    return path, delta, phi, max_path_prob