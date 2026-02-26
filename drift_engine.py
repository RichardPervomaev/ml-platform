import numpy as np

class DriftEngine:

    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def calculate_psi(self, expected, actual, bins=10):

        expected_counts, bin_edges = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_perc = expected_counts / len(expected)
        actual_perc = actual_counts / len(actual)

        psi = np.sum(
            (expected_perc - actual_perc) *
            np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
        )

        return psi

    def detect(self, train_data, prod_data):

        psi = self.calculate_psi(train_data, prod_data)

        if psi > self.threshold:
            return True, psi

        return False, psi


drift_engine = DriftEngine()
