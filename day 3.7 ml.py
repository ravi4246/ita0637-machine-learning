import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target
# (a) Print the first five rows
print("First five rows of the dataset:")
print(df.head())
# (b) Basic statistical computations
print("\nBasic statistical computations:")
print(df.describe())
# (c) Print the columns and their data types
print("\nColumns and their data types:")
print(df.dtypes)
# (d) Detect and handle null values (if any)
print("\nDetecting null values:")
print(df.isnull().sum())
# If there are any null values, replace them with the mode
for column in df.columns:
    if df[column].isnull().sum() > 0:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)
print("\nNull values after handling:")
print(df.isnull().sum())
# (e) Explore the dataset using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()
# (f) Split the data into test and train sets
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# (g) Train a Linear Regression model and predict the price of a house
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
example_house = np.array([[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]]) # example values
predicted_price = model.predict(example_house)
print("\nPredicted price for the example house:", predicted_price[0])
