import pandas as pd
data = {
    'Size': ['Big', 'Small', 'Small', 'Big', 'Small'],
    'Color': ['Red', 'Red', 'Red', 'Blue', 'Blue'],
    'Shape': ['Circle', 'Triangle', 'Circle', 'Circle', 'Circle'],
    'Class': ['No', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)
def find_s(dataset):
    hypothesis = ['0'] * (len(dataset.columns) - 1)
    for i, row in dataset.iterrows():
        if row[-1] == 'Yes': 
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = row[j]
                elif hypothesis[j] != row[j]:
                    hypothesis[j] = '?'
    return hypothesis
specific_hypothesis = find_s(df)
print("Most Specific Hypothesis:", specific_hypothesis)
