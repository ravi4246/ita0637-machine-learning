import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 100)
X = X[:, np.newaxis]
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred_linear = linear_regressor.predict(X)
polynomial_features = PolynomialFeatures(degree=3)
X_poly = polynomial_features.fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)
mse_linear = mean_squared_error(y, y_pred_linear)
mse_poly = mean_squared_error(y, y_pred_poly)
print(f"Mean Squared Error for Linear Regression: {mse_linear:.2f}")
print(f"Mean Squared Error for Polynomial Regression (degree=3): {mse_poly:.2f}")
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, y_pred_linear, color='green', label='Linear Regression')
plt.plot(X, y_pred_poly, color='red', label='Polynomial Regression (degree=3)')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
