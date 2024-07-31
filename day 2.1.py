import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
np.random.seed(42)
n_samples = 100
data = {
    'Make': np.random.choice(['Toyota', 'Ford', 'BMW', 'Honda'], n_samples),
    'Model': np.random.choice(['Model A', 'Model B', 'Model C', 'Model D'], n_samples),
    'Year': np.random.randint(2000, 2022, n_samples),
    'Engine_Size': np.random.uniform(1.0, 4.0, n_samples),
    'Doors': np.random.choice([2, 4], n_samples),
    'Sale_Price': np.random.uniform(15000, 50000, n_samples)
}
car_data = pd.DataFrame(data)
print("First five rows of the dataset:")
print(car_data.head())
print("\nStatistical summary of the dataset:")
print(car_data.describe())
print("\nColumns and their data types:")
print(car_data.dtypes)
car_data.loc[car_data.sample(frac=0.1).index, 'Engine_Size'] = np.nan
print("\nNull values before handling:")
print(car_data.isnull().sum())
mode_value = car_data['Engine_Size'].mode()[0]
car_data['Engine_Size'].fillna(mode_value, inplace=True)
print("\nNull values after handling:")
print(car_data.isnull().sum())
plt.figure(figsize=(10, 6))
sns.heatmap(car_data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Car Data Correlations')
plt.show()
car_data = pd.get_dummies(car_data, drop_first=True)
X = car_data.drop('Sale_Price', axis=1)
y = car_data['Sale_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.astype(int))
y_pred = nb_model.predict(X_test)
accuracy = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error: {accuracy:.2f}%")
