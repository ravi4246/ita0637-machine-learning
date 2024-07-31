import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier()
}
results = {}
for name, model in models.items():
    start_time = time.time() 
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)  
    end_time = time.time() 
    accuracy = accuracy_score(y_test, y_pred)
    execution_time = end_time - start_time
    results[name] = {
        "Accuracy": accuracy,
        "Execution Time (s)": execution_time
    }
print(f"{'Model':<25} {'Accuracy':<10} {'Execution Time (s)':<20}")
print("-" * 55)
for name, metrics in results.items():
    print(f"{name:<25} {metrics['Accuracy']:<10.2f} {metrics['Execution Time (s)']:<20.4f}")
