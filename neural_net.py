import numpy as np
from sklearn.neural_network import MLPClassifier

# Load the data
data_train = np.load('data_train.npy', allow_pickle=True)
data_test = np.load('data_test.npy', allow_pickle=True)
labels_train = np.genfromtxt('label_train.csv', delimiter=',', skip_header=1)
vocab_map = np.load('vocab_map.npy', allow_pickle=True)

# Use the second column of labels_train as the actual labels and reshape them to 1D array
labels_train = labels_train[:, 1]

# Initialize the MLP neural network classifier
neural_net = MLPClassifier(alpha=1, max_iter=1000, random_state=42)

# Fit the classifier to the training data
neural_net.fit(data_train, labels_train)
print("fitted")
print("predicting")
# Predict the labels for data_test
predicted_labels = neural_net.predict(data_test)
ids = np.arange(0, len(predicted_labels))
output_data = np.column_stack((ids, predicted_labels))

print("predicted")

# Save the predicted labels to a file (e.g., in NumPy format)
np.save('predicted_labels.npy', output_data)

# Alternatively, save to a CSV file
np.savetxt('predicted_labels.csv', output_data, delimiter=',', fmt=['%d', '%d'], header='ID,label')
print("saved")