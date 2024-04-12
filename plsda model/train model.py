from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pyopls import OPLS
from joblib import dump
import pandas as pd

# Load the data
spectra = pd.read_csv('30train full dataset.csv')
target = spectra.CLASS
spectra = spectra.drop(['NAME','CLASS','SAMPLE'], axis=1)

# Initialize OPLS model
opls = OPLS(n_components=20)

# Transform the data using OPLS
X_opls = opls.fit_transform(spectra, target)

# Define classifiers
classifiers = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}

# Specify hyperparameter grids for each classifier
param_grids = {
    'DecisionTree': {'max_depth': [None, 15, 25]},
    'RandomForest': {'n_estimators': [41], 'max_depth': [None, 10, 20]},
    'GradientBoosting': {'n_estimators': [15], 'learning_rate': [0.5]},
    'KNN': {'n_neighbors': [3]},
    'SVM': {'C': [0.15], 'gamma': ['scale','auto']}
}

# User selection for model
selected_model = 'SVM'  # Change this to select a different model

# Initialize selected classifier
classifier = classifiers[selected_model]

# Perform grid search with cross-validation
grid_search = GridSearchCV(classifier, param_grids[selected_model], cv=15, scoring='accuracy')
grid_search.fit(X_opls, target)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Initialize the best classifier with the best hyperparameters
best_classifier = classifiers[selected_model].set_params(**grid_search.best_params_)

# Perform cross-validation with the best classifier
cv_scores = cross_val_score(best_classifier, X_opls, target, cv=15, scoring='accuracy')

# Calculate mean cross-validation score
mean_cv_score = cv_scores.mean()
print(f"Mean Cross-Validation Accuracy: {mean_cv_score}")

# Make cross-validated predictions with the best classifier
cv_predictions = cross_val_predict(best_classifier, X_opls, target, cv=15)

# Calculate additional metrics
precision = precision_score(target, cv_predictions)
recall = recall_score(target, cv_predictions)
f1 = f1_score(target, cv_predictions)
conf_matrix = confusion_matrix(target, cv_predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Train the best classifier on the entire dataset
best_classifier.fit(X_opls, target)

# Save the OPLS model and the best classifier
dump(opls, 'opls_model.joblib')
dump(best_classifier, f'{selected_model}_best_classifier_model.joblib')
