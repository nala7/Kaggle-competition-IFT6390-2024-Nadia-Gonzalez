import numpy as np

# data_train = np.load('data_train.npy', allow_pickle=True)
# data_test = np.load('data_test.npy', allow_pickle=True)
# labels_train = np.genfromtxt('label_train.csv', delimiter=',', skip_header=1)
# vocab_map = np.load('vocab_map.npy', allow_pickle=True)
#
# labels_train = labels_train[:, 1]

class MyLogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(z)
        return A

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            A = self.feed_forward(X)
            self.losses.append(self.compute_loss(y, A))
            dz = A - y  # derivative of sigmoid and bce X.T*(A-y)
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        threshold = .5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def score(self, X, y):
        # Generate predictions
        y_pred = self.predict(X)

        # Calculate accuracy
        accuracy = np.mean(y == y_pred)
        return accuracy


# logistic_regression = LogisticRegression()
# print("Logistic Regression starting...")
# logistic_regression.fit(data_train, labels_train)
# print("Fitted")
# print("Predicting...")
# y_predicted = logistic_regression.predict(data_test)
# ids = np.arange(0, len(y_predicted))
# output_data = np.column_stack((ids, y_predicted))
# print("predicted")
#
# # Save the predicted labels to a file (e.g., in NumPy format)
# np.save('lr_predicted_labels.npy', output_data)
#
# # Alternatively, save to a CSV file
# np.savetxt('lr_predicted_labels.csv', output_data, delimiter=',', fmt=['%d', '%d'], header='ID,label')
# print("saved")


