from scipy.sparse import lil_matrix
import numpy as np


class ChainModel:
    def __init__(self):
        self.proba_matrix = None

    def fit(self, encoded_features):
        encoded_features = np.asarray(encoded_features, dtype=int)
        size = np.max(encoded_features) + 1
        self.proba_matrix = lil_matrix((size, size), dtype=float)
        for curr_state, next_state in encoded_features:
            self.proba_matrix[curr_state, next_state] += 1
        for row in range(self.proba_matrix.shape[0]):
            if self.proba_matrix[row].count_nonzero() != 0:
                div = self.proba_matrix[row].sum()
                for col in self.proba_matrix[row].nonzero()[1]:
                    self.proba_matrix[row, col] /= div

    def predict_next(self, state):
        indices = self.proba_matrix[state].nonzero()[1]
        p = self.proba_matrix[state, indices].toarray().ravel()
        return np.random.choice(indices, 1, p=p)[0]

    def predict(self, count, initial):
        result = []
        for _ in range(count):
            result.append(initial)
            try:
                initial = self.predict_next(initial)
            except ValueError:
                print("can't generate > {}, error".format(len(result)))
                return result
        return result
