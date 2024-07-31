import pandas as pd
data = pd.DataFrame({
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': ['1980', '1970', '1990', '1980', '1980'],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
})
print("Dataset:")
print(data)
features = data.columns[:-1]
target = data.columns[-1]
S = ['0'] * len(features)
G = [['?'] * len(features)]
def more_general(h1, h2):
    """ Returns True if hypothesis h1 is more general than or equal to hypothesis h2 """
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == '?' or (x != '0' and (x == y or y == '0'))
        more_general_parts.append(mg)
    return all(more_general_parts)
def min_generalizations(h, x):
    """ Returns the minimal generalizations of hypothesis h to cover example x """
    h_new = list(h)
    for i in range(len(h)):
        if not more_general([h[i]], [x[i]]):
            h_new[i] = '?' if h[i] != '0' else x[i]
    return [tuple(h_new)]
def min_specializations(h, domains, x):
    """ Returns the minimal specializations of hypothesis h to exclude example x """
    results = []
    for i in range(len(h)):
        if h[i] == '?':
            for val in domains[i]:
                if val != x[i]:
                    h_new = h[:i] + [val] + h[i+1:]
                    results.append(h_new)
        elif h[i] != '0':
            h_new = h[:i] + ['0'] + h[i+1:]
            results.append(h_new)
    return results
domains = {i: list(set(data[features[i]])) for i in range(len(features))}
for index, row in data.iterrows():
    x = row[:-1]
    y = row[-1]
    if y == 'Positive':
        G = [g for g in G if more_general(g, x)]
        for s in S:
            if not more_general(x, s):
                S = min_generalizations(s, x)
    else:
        S = [s for s in S if not more_general(s, x)]
        all_specializations = []
        for g in G:
            all_specializations += min_specializations(g, domains, x)
        G = all_specializations

    G = [g for g in G if any(more_general(g, s) for s in S)]
    S = [s for s in S if any(more_general(g, s) for g in G)]
print("\nFinal S:", S)
print("Final G:", G)
