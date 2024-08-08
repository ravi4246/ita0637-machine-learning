import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
data = {
    'Age': [25, 30, 45, 35, 50, 23, 34, 65, 30, 40],
    'Income': [50000, 60000, 80000, 70000, 120000, 45000, 65000, 100000, 55000, 85000],
    'LoanAmount': [20000, 15000, 30000, 25000, 40000, 10000, 20000, 35000, 15000, 30000],
    'Occupation': ['Engineer', 'Teacher', 'Doctor', 'Engineer', 'Lawyer', 'Engineer', 'Nurse', 'Retired', 'Teacher', 'Doctor'],
    'CreditScore': [650, 700, 750, 720, 780, 620, 690, 800, 680, 740]
}
df = pd.DataFrame(data)
print("First five rows of the dataset:")
print(df.head())
print("\nBasic statistical summary:")
print(df.describe())
print("\nColumns and data types:")
print(df.dtypes)
print("\nChecking for null values:")
print(df.isnull().sum())
plt.figure(figsize=(10, 6))
sns.boxplot(x='Occupation', y='CreditScore', data=df)
plt.title('Credit Scores Based on Occupation')
plt.show()
df['Occupation'] = df['Occupation'].astype('category').cat.codes
X = df.drop('CreditScore', axis=1)
y = df['CreditScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
