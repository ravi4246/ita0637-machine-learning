import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_reg_train, y_reg_train)
y_reg_pred = linear_model.predict(X_reg_test)
logistic_model = LogisticRegression()
logistic_model.fit(X_clf_train, y_clf_train)
y_clf_pred = logistic_model.predict(X_clf_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)
accuracy = accuracy_score(y_clf_test, y_clf_pred)
cm = confusion_matrix(y_clf_test, y_clf_pred)
report = classification_report(y_clf_test, y_clf_pred)
print("Linear Regression Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print("\nLogistic Regression Performance:")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(report)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_reg_test, y_reg_test, color='blue', label='Actual')
plt.plot(X_reg_test, y_reg_pred, color='red', label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(X_clf_test[:, 0], X_clf_test[:, 1], c=y_clf_pred, cmap='coolwarm', edgecolors='k', label='Predicted')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression')
plt.legend()
plt.tight_layout()
plt.show()
