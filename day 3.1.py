from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print(data.head())
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=data, palette='viridis')
plt.title('Sepal Width vs Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(loc='upper right')
plt.show()
X = data.drop('species', axis=1) 
y = data['species']  
y = pd.Categorical(y).codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
new_sample = [[5, 3, 1, 0.3]]
predicted_species_code = model.predict(new_sample)
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_species = species_mapping[predicted_species_code[0]]
print(f"Predicted species for the sample {new_sample}: {predicted_species}")
