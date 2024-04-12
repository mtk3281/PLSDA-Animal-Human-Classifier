from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import load
import pandas as pd

# Load the saved OPLS model and classifier
opls_model = load('opls_model.joblib')
classifier_model = load("SVM_best_classifier_model.joblib")

# Load the new test data
new_test_data = pd.read_csv('30test full dataset.csv')

# Preprocess the new test data using the loaded OPLS model
new_test_data_filtered = new_test_data[new_test_data['NAME'].isin(['HUMAN', 'ANIMAL'])]  # Assuming 'NAME' is the column name for categories
X_new_test_transformed = opls_model.transform(new_test_data_filtered.drop(columns=['NAME', 'CLASS','SAMPLE']))  # Drop non-numeric columns

# Separate features and target labels
y_new_test = new_test_data_filtered['CLASS']  # Assuming 'CLASS' is the column name for target labels

# Make predictions on the new test data
y_pred_new_test = classifier_model.predict(X_new_test_transformed)

# Calculate accuracy
accuracy_new_test = accuracy_score(y_new_test, y_pred_new_test)
print(f"Accuracy on New Test Data: {accuracy_new_test}")

# Other metrics
conf_matrix = confusion_matrix(y_new_test, y_pred_new_test)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_new_test, y_pred_new_test)
print("Classification Report:")
print(class_report)
