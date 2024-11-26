import numpy as np
import matplotlib.pyplot as plt

class MySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def plot_decision_boundary(self, X, y):
        plt.figure(figsize=(10, 6))

        # Scatter plot of data points
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7, edgecolor='k')

        # Define x-axis range for decision boundary
        x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x0 = np.linspace(x0_min, x0_max, 100)

        # Compute decision boundary
        x1 = -(self.w[0] * x0 + self.b) / self.w[1]
        plt.plot(x0, x1, 'k--', linewidth=2, label="Decision Boundary")

        # Plot margins
        margin = 1 / np.sqrt(np.sum(self.w ** 2))
        x1_margin_pos = x1 + margin
        x1_margin_neg = x1 - margin
        plt.plot(x0, x1_margin_pos, 'k:', linewidth=1, label="Margin")
        plt.plot(x0, x1_margin_neg, 'k:', linewidth=1)

        # Set plot limits and labels
        plt.xlim(x0_min, x0_max)
        plt.ylim(x1_min, x1_max)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.title("SVM Decision Boundary with Data Points")
        plt.show()

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy