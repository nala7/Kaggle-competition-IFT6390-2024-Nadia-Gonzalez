import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class MyKNN:
    def __init__(self, k=3):
        self.k = k  # Number of neighbors to consider

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x, idx) for idx, x in enumerate(X_test)]

        return np.array(predictions)

    def _predict(self, x, i):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        return accuracy

    def plot_training_data(self):
        plt.figure(figsize=(8, 6))

        for label in np.unique(self.y_train):
            plt.scatter(self.X_train[self.y_train == label][:, 0], self.X_train[self.y_train == label][:, 1],
                        label=f"Class {label}", s=100, alpha=0.6)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Binary Training Data Plot")
        plt.legend()
        plt.show()