import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True).frame
data.rename(columns={'MedInc': 'Income', 'MedHouseVal': 'HouseValue'}, inplace=True)
print("First five rows of the dataset:")
print(data.head())
print("\nBasic Statistical Computations:")
print(data.describe())
print("\nColumns and Data Types:")
print(data.dtypes)
sns.pairplot(data, x_vars=['Income'], y_vars='HouseValue', height=5, aspect=0.7, kind='scatter')
plt.title('House Value vs Income')
plt.show()
print("\nChecking for null values:")
print(data.isnull().sum())
data.fillna(data.mode().iloc[0], inplace=True)
print("\nNull values after filling:")
print(data.isnull().sum())
X = data[['Income']]
y = data['HouseValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Income')
plt.ylabel('House Value')
plt.title('Linear Regression: House Value vs Income')
plt.show()
