import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 100)
X = X[:, np.newaxis]
polynomial_features = PolynomialFeatures(degree=3)
X_poly = polynomial_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
plt.scatter(X, y, color='blue', s=10, label='Original data')
plt.plot(X, y_pred, color='red', label='Polynomial regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
