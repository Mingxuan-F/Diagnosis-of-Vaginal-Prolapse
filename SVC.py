# Import required libraries
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from pandas import read_csv

# Assuming you have your data in the form of X (features) and y (labels/targets)
filename = 'GAIN_SCAD_selected.csv'
data = read_csv(filename,delimiter=',')
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]
# Lists to store results for each iteration
accuracy_list = []
f1_list = []
roc_auc_list = []

# Repeat the experiment 100 times
for _ in range(100):
    # Step 1: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Initialize the SVM Classifier
    # You can adjust hyperparameters like C (regularization parameter) and kernel to your preference
    svm_classifier = SVC(C=1.0, kernel='linear', probability=True, random_state=42)

    # Step 3: Train the SVM Classifier on the training data
    svm_classifier.fit(X_train, y_train)

    # Step 4: Make predictions on the test data
    y_pred = svm_classifier.predict(X_test)

    # Step 5: Calculate probabilities for AUC
    y_probs = svm_classifier.predict_proba(X_test)[:, 1]

    # Step 6: Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)

    # Store the results for this iteration
    accuracy_list.append(accuracy)
    f1_list.append(f1)
    roc_auc_list.append(roc_auc)

# Calculate the mean and variance of performance metrics
mean_accuracy = np.mean(accuracy_list)
variance_accuracy = np.var(accuracy_list)

mean_f1 = np.mean(f1_list)
variance_f1 = np.var(f1_list)

mean_roc_auc = np.mean(roc_auc_list)
variance_roc_auc = np.var(roc_auc_list)

# Print the results
print("Mean Accuracy:", mean_accuracy)
print("Variance of Accuracy:", variance_accuracy)

print("Mean F1 Score:", mean_f1)
print("Variance of F1 Score:", variance_f1)

print("Mean AUC:", mean_roc_auc)
print("Variance of AUC:", variance_roc_auc)
