import pomegranate as pg
import numpy as np

class Distributions(object):

    def __init__(self, train):
        self.train = train
        self.n_features = train.shape[1]-1
        self.data_columns = list(train.columns.values)
        self.hidden_sequence = train['hypnogram_User'].tolist()
        for i in range(0, len(self.hidden_sequence)):
            if self.hidden_sequence[i] == "NotScored":
                self.hidden_sequence[i] = self.hidden_sequence[i - 1]

    def gauss_kernel_dist(self, feature_names):
        possible_states = np.unique(['NonREM1', "NonREM2", "NonREM3", "REM", "Wake"]).tolist()
        state_names = []
        set_of_state_sets = []
        for state in range(0, len(possible_states)):
            if not self.train[self.train.hypnogram_User == possible_states[state]].empty:
                set_of_state_sets.append(self.train[self.train.hypnogram_User == possible_states[state]])
                state_names.append(possible_states[state])
        binary_features = ["Gain", "Bradycardia", "LegMovement", "CentralApnea", "Arousal", "Hypopnea",
                           "RelativeDesaturation", "Snore", "ObstructiveApnea", "MixedApnea", "LongRR", "Tachycardia"]
        state_multidistributions = []
        for set in range(0, len(set_of_state_sets)):
            state_dist = []
            for i in range(0, self.n_features):
                if feature_names[i] in binary_features:
                    state_dist.append(
                        pg.BernoulliDistribution.from_samples(set_of_state_sets[set][self.data_columns[i]].tolist()))
                else:
                    state_dist.append(
                        pg.GaussianKernelDensity.from_samples(set_of_state_sets[set][self.data_columns[i]].tolist()))
            state_multidistributions.append(state_dist)
        dist = [pg.IndependentComponentsDistribution(x) for x in state_multidistributions]

        return dist, state_names

    def normal_dist(self, feature_names):
        possible_states = np.unique(['NonREM1', "NonREM2", "NonREM3", "REM", "Wake"]).tolist()
        state_names = []
        set_of_state_sets = []
        for state in range(0, len(possible_states)):
            if not self.train[self.train.hypnogram_User == possible_states[state]].empty:
                set_of_state_sets.append(self.train[self.train.hypnogram_User == possible_states[state]])
                state_names.append(possible_states[state])
        binary_features = ["Gain", "Bradycardia", "LegMovement", "CentralApnea", "Arousal", "Hypopnea",
                           "RelativeDesaturation", "Snore", "ObstructiveApnea", "MixedApnea", "LongRR", "Tachycardia"]
        state_multidistributions = []
        for set in range(0, len(set_of_state_sets)):
            state_dist = []
            for i in range(0, self.n_features):
                if feature_names[i] in binary_features:
                    state_dist.append(
                        pg.BernoulliDistribution.from_samples(set_of_state_sets[set][self.data_columns[i]].tolist()))
                else:
                    state_dist.append(
                        pg.NormalDistribution.from_samples(set_of_state_sets[set][self.data_columns[i]].tolist()))
            state_multidistributions.append(state_dist)
        dist = [pg.IndependentComponentsDistribution(x) for x in state_multidistributions]

        return dist, state_names