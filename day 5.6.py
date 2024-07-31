import numpy as np
data = [
    ['Some', 'Small', 'No', 'Affordable', 'Few', 'No'],
    ['Many', 'Big', 'No', 'Expensive', 'Many', 'Yes'],
    ['Many', 'Medium', 'No', 'Expensive', 'Few', 'Yes'],
    ['Many', 'Small', 'No', 'Affordable', 'Many', 'Yes']
]
features = np.array([row[:-1] for row in data])
target = np.array([row[-1] for row in data])
def find_s(features, target):
    hypothesis = ['0'] * len(features[0])
    for i, example in enumerate(features):
        if target[i] == 'Yes':
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = example[j]
                elif hypothesis[j] != example[j]:
                    hypothesis[j] = '?'
    return hypothesis
specific_hypothesis = find_s(features, target)
print("The most specific hypothesis is:", specific_hypothesis)
