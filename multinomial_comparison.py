from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np


train_size = 1000
test_size = 200


data_train = np.load('data_train.npy', allow_pickle=True)[:train_size]  # Limit to 500 samples
data_test = np.load('data_test.npy', allow_pickle=True)[:test_size]  # Limit to 100 samples
labels_train = np.genfromtxt('label_train.csv', delimiter=',', skip_header=1)[:train_size, 1]  # Limit to 500 labels

labels_train = labels_train.astype(int)


# Custom Multinomial Logistic Regression
class MyMultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=50, batch_size=128):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.random.randn(1, n_classes) * 0.01

        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        for i in range(self.n_iters):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y_one_hot = y_one_hot[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X[start:end]
                y_batch = y_one_hot[start:end]

                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._softmax(linear_model)

                dw = (1 / X_batch.shape[0]) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / X_batch.shape[0]) * np.sum(y_pred - y_batch, axis=0, keepdims=True)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._softmax(linear_model)
        return np.argmax(y_pred, axis=1)


# Evaluation function
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1


# Prepare data
scaler = StandardScaler()
data_train_scaled = scaler.fit_transform(data_train)
data_test_scaled = scaler.transform(data_test)

# Define hyperparameter combinations
hyperparams = [
    {"learning_rate": 0.01, "n_iters": 500, "batch_size": 128},
    {"learning_rate": 0.01, "n_iters": 1000, "batch_size": 128},
    {"learning_rate": 0.05, "n_iters": 500, "batch_size": 128},
    {"learning_rate": 0.05, "n_iters": 1000, "batch_size": 128},
    {"learning_rate": 0.01, "n_iters": 500, "batch_size": 64},
    {"learning_rate": 0.01, "n_iters": 1000, "batch_size": 64},
    {"learning_rate": 0.05, "n_iters": 500, "batch_size": 64},
    {"learning_rate": 0.05, "n_iters": 1000, "batch_size": 64},
]

# Train and evaluate models with different hyperparameters
results = {
    "Learning Rate": [],
    "Iterations": [],
    "Batch Size": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

for params in hyperparams:
    model = MyMultinomialLogisticRegression(
        learning_rate=params["learning_rate"],
        n_iters=params["n_iters"],
        batch_size=params["batch_size"]
    )
    print(f"Training with {params}...")
    model.fit(data_train_scaled, labels_train)

    y_pred = model.predict(data_test_scaled)
    accuracy, precision, recall, f1 = evaluate_model(labels_train[:test_size], y_pred)

    results["Learning Rate"].append(params["learning_rate"])
    results["Iterations"].append(params["n_iters"])
    results["Batch Size"].append(params["batch_size"])
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)

# Display results
results_df = pd.DataFrame(results)
print("Model Comparison Results with Hyperparameters")
print(results_df)
