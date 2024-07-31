import pandas as pd
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': ['1980', '1970', '1990', '1980', '1980'],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example_Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}
df = pd.DataFrame(data)
hypothesis = ['0', '0', '0', '0', '0']
for i, row in df.iterrows():
    if row['Example_Type'] == 'Positive':
        for j in range(len(hypothesis)):
            if hypothesis[j] == '0':
                hypothesis[j] = row[j]
            elif hypothesis[j] != row[j]:
                hypothesis[j] = '?'
print("The most specific hypothesis is:", hypothesis)
