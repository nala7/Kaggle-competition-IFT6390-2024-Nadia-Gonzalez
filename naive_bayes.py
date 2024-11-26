import numpy as np

class MyNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)

        for c in self.classes:
            # Filter samples of the current class
            X_c = X_train[y_train == c]

            # Calculate mean, variance, and prior for the current class
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X_train.shape[0]

    def predict(self, X_test):
        predictions = [self._predict(x, idx) for idx, x in enumerate(X_test)]
        return np.array(predictions)

    def _predict(self, x, idx):
        # Calculate posterior probability for each class
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(self._pdf(c, x)))
            posterior = prior + conditional
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        # Apply Gaussian PDF formula
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def score(self, X_test, y_test):
        """Calculate the accuracy of the model on the test data."""
        predictions = self.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        return accuracy
