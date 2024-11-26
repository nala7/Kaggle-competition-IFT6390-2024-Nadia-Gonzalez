# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, \
#     StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import train_test_split
#
# # Load a smaller sample of the data for testing
# data_train = np.load('data_train.npy', allow_pickle=True)[:1000]  # Use the first 1000 samples
# # data_test = np.load('data_test.npy', allow_pickle=True)[:200]  # Use the first 200 samples
# labels_train = np.genfromtxt('label_train.csv', delimiter=',', skip_header=1)[:1000]  # First 1000 labels
#
# # Ensure labels are a 1D array
# if labels_train.ndim == 2 and labels_train.shape[1] > 1:
#     labels_train = labels_train[:, 0]
# labels_train = labels_train.astype(int)
#
# # Split data for training and testing
# X_train, X_val, y_train, y_val = train_test_split(data_train, labels_train, test_size=0.2, random_state=42)
# y_train = y_train.ravel()
# y_val = y_val.ravel()
#
# print("X_train shape:", X_train.shape)
# print("X_val shape:", X_val.shape)
# print("y_train shape:", y_train.shape)
# print("y_val shape:", y_val.shape)
# print("Unique labels in y_train:", np.unique(y_train))
# print("Unique labels in y_val:", np.unique(y_val))
#
#
# # Dictionary to hold evaluation results
# results = {
#     "Model": [],
#     "Accuracy": [],
#     "Precision": [],
#     "Recall": [],
#     "F1 Score": []
# }
#
#
# # Helper function to evaluate and store results
# def evaluate_model(model, model_name):
#     print("Fitting...")
#     model.fit(X_train, y_train)
#     print("Predicting...")
#     y_pred = model.predict(X_val)
#     accuracy = accuracy_score(y_val, y_pred)
#     print("{} Accuracy: {}".format(model_name, accuracy))
#     precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
#     recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
#     f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
#
#     results["Model"].append(model_name)
#     results["Accuracy"].append(accuracy)
#     results["Precision"].append(precision)
#     results["Recall"].append(recall)
#     results["F1 Score"].append(f1)
#
#
# # 1. Gradient Boosting
# print("Gradient Boosting")
# gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
# evaluate_model(gb_model, "Gradient Boosting")
#
# # 2. Random Forests
# rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
# evaluate_model(rf_model, "Random Forest")
#
# # 3. Bagging with Decision Tree
# # bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
# bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
# evaluate_model(bagging_model, "Bagging (Decision Tree)")
#
# # 4. Voting Classifier (Hard Voting)
# voting_model = VotingClassifier(
#     estimators=[
#         ('lr', LogisticRegression()),
#         ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#         ('svc', SVC(kernel='linear', probability=True, random_state=42))
#     ], voting='hard')
# evaluate_model(voting_model, "Voting Classifier (Hard)")
#
# # # 5. Stacking Classifier
# # stacking_model = StackingClassifier(
# #     estimators=[
# #         ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
# #         ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42))
# #     ], final_estimator=LogisticRegression(), cv=1
# # )
# # evaluate_model(stacking_model, "Stacking Classifier")
#
# # Display the results
# results_df = pd.DataFrame(results)
# print("Model Comparison Results")
# print(results_df)
