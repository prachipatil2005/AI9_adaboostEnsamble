import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, Y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Define the base classifier
base_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize the AdaBoost classifier
ada_boost = AdaBoostClassifier(base_classifier, n_estimators=50, algorithm='SAMME', random_state=42)

# Fit the model
ada_boost.fit(X_train, Y_train)

# Make predictions
Y_pred = ada_boost.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)

# Print accuracy
print(f"Accuracy of AdaBoost classifier: {accuracy:.2f}")


#pip install scikit-learn
