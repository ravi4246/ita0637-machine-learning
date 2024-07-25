# Define the dataset
data = [
    ['Circular', 'Large', 'Light', 'Smooth', 'Thick', 'Malignant'],
    ['Circular', 'Large', 'Light', 'Irregular', 'Thick', 'Malignant'],
    ['Oval', 'Large', 'Dark', 'Smooth', 'Thin', 'Benign'],
    ['Oval', 'Large', 'Light', 'Irregular', 'Thick', 'Malignant']
]
hypothesis = ['0', '0', '0', '0', '0']
for example in data:
    if example[-1] == 'Malignant':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = example[i]
            elif hypothesis[i] != example[i]:
                hypothesis[i] = '?'
print("The most specific hypothesis found by Find-S algorithm:")
print(hypothesis)
