import numpy as np

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def compute_precision(y_true, y_pred):
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positive = np.sum(y_pred == 1)
    if predicted_positive == 0:
        return 0
    return true_positive / predicted_positive


def compute_recall(y_true, y_pred):
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    actual_positive = np.sum(y_true == 1)
    if actual_positive == 0:
        return 0
    return true_positive / actual_positive


def compute_f1_score(y_true, y_pred):
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)