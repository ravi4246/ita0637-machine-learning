import pandas as pd
data = {
    'Example': [1, 2, 3, 4],
    'Shape': ['Circular', 'Circular', 'Oval', 'Oval'],
    'Size': ['Large', 'Large', 'Large', 'Large'],
    'Color': ['Light', 'Light', 'Dark', 'Light'],
    'Surface': ['Smooth', 'Irregular', 'Smooth', 'Irregular'],
    'Thickness': ['Thick', 'Thick', 'Thin', 'Thick'],
    'Target Concept': ['Malignant (+)', 'Malignant (+)', 'Benign (-)', 'Malignant (+)']
}
df = pd.DataFrame(data)
features = df.drop(['Example', 'Target Concept'], axis=1).values
target = df['Target Concept'].values
specific_h = ['0'] * len(features[0])
general_h = [['?' for _ in range(len(features[0]))] for _ in range(len(features[0]))]
for i, instance in enumerate(features):
    if target[i] == 'Malignant (+)':
        for j in range(len(specific_h)):
            if specific_h[j] == '0':
                specific_h[j] = instance[j]
            elif specific_h[j] != instance[j]:
                specific_h[j] = '?'
        
        for j in range(len(general_h)):
            if general_h[j][j] != '?' and general_h[j][j] != instance[j]:
                general_h[j][j] = '?'
    else:
        for j in range(len(general_h)):
            if general_h[j][j] == '?' or specific_h[j] == instance[j]:
                general_h[j][j] = '?'
            else:
                general_h[j][j] = specific_h[j]
general_h = [g for g in general_h if g != ['?' for _ in range(len(features[0]))]]
print("Most Specific Hypothesis:", specific_h)
print("Most General Hypotheses:", general_h)
