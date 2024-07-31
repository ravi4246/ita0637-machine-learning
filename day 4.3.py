import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 100)
X = X[:, np.newaxis]
y = y[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
model = LinearRegression()
model.fit(X_poly_train, y_train)
y_train_pred = model.predict(X_poly_train)
y_test_pred = model.predict(X_poly_test)
print("Mean Squared Error (Train):", mean_squared_error(y_train, y_train_pred))
print("R2 Score (Train):", r2_score(y_train, y_train_pred))
print("Mean Squared Error (Test):", mean_squared_error(y_test, y_test_pred))
print("R2 Score (Test):", r2_score(y_test, y_test_pred))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(poly.transform(X)), color='red', label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression')
plt.show()
