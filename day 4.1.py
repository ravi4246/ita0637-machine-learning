from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print("First five rows of the dataset:")
print(df.head())
print("\nBasic Statistical Computations:")
print(df.describe())
print("\nColumns and Data Types:")
print(df.dtypes)
print("\nChecking for null values:")
print(df.isnull().sum())
df.fillna(df.mode().iloc[0], inplace=True)
print("\nNull values after filling (if any):")
print(df.isnull().sum())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
