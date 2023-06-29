import multiprocessing as mp

from EVM import MultipleEVM
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder


class EVMWrapper:
    """
    Wrapper for original EVM (https://bitbucket.org/vastlab/evm).

    Is used for optional unittests, to verify that the TensorFlow EVM
    generates comparable results to the original implementation.
    """

    def __init__(self, tail_size, distance_multiplier=0.5, cover_threshold=None,
                 distance_function="euclidean", parallel=None):
        if distance_function not in ["euclidean"]:
            raise RuntimeError("EVMWrapper does not know given distance '{}'.".format(distance_function))

        self.enc = LabelEncoder()
        self.evm = MultipleEVM(
            tailsize=tail_size, distance_multiplier=distance_multiplier, cover_threshold=cover_threshold,
            distance_function=euclidean)

        self.parallel = parallel
        if parallel == 0:
            self.parallel = mp.cpu_count()

    def fit(self, x, y):
        y = self.enc.fit_transform(y)

        positive_classes = []
        for cl in np.unique(y):
            positive_classes.append(x[y == cl])

        self.evm.train(positive_classes, parallel=self.parallel)

    def predict(self, x):
        _, y_pred_and_idx = self.evm.max_probabilities(x, parallel=self.parallel)
        return self.enc.inverse_transform(np.array([t[0] for t in y_pred_and_idx]))

    def predict_proba(self, x):
        result = self.evm.probabilities(x, parallel=self.parallel)
        return np.array([[np.max(b) for b in a] for a in result])

    def extreme_vectors(self):
        ev_x = []
        ev_y = []
        for i, evm in enumerate(self.evm.evms):
            ev_x.append(evm.extreme_vectors)
            ev_y.append(np.full((len(evm.extreme_vectors),), self.enc.classes_[i]))

        return np.concatenate(ev_x, axis=0), np.concatenate(ev_y, axis=0)
