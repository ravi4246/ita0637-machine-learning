import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
file_path = "D:\Slot-D\ML Practical\day 1.7.csv"
data = pd.read_csv(file_path)
print("First five rows of the dataset:")
print(data.head())
print("\nBasic statistical summary:")
print(data.describe())
print("\nColumns and their data types:")
print(data.dtypes)
print("\nNull values in the dataset:")
print(data.isnull().sum())
for column in data.columns:
    if data[column].isnull().sum() > 0:
        mode_value = data[column].mode()[0]
        data[column].fillna(mode_value, inplace=True)
print("\nNull values after replacement:")
print(data.isnull().sum())
plt.figure(figsize=(10, 6))
sns.boxplot(x='Occupation', y='Credit_Score', data=data)
plt.title('Credit Scores Based on Occupation')
plt.xticks(rotation=45)
plt.show()
X = data.drop('Credit_Score', axis=1)
y = data['Credit_Score']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
