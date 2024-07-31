import pandas as pd
data = pd.DataFrame({
    'Citations': ['Some', 'Many', 'Many', 'Many'],
    'Size': ['Small', 'Big', 'Medium', 'Small'],
    'In Library': ['No', 'No', 'No', 'No'],
    'Price': ['Affordable', 'Expensive', 'Expensive', 'Affordable'],
    'Editions': ['Few', 'Many', 'Few', 'Many'],
    'Buy': ['No', 'Yes', 'Yes', 'Yes']
})
def candidate_elimination(data):
    def initialize_hypothesis(data):
        return ['0'] * (len(data.columns) - 1)
    def is_more_general(h1, h2):
        for x, y in zip(h1, h2):
            if x != '?' and (x != y and y != '?'):
                return False
        return True
    def generalize_S(S, instance):
        for i in range(len(S)):
            if not instance[i] == S[i]:
                S[i] = '?'
        return S
    def specialize_G(G, instance, domains):
        G_new = []
        for g in G:
            for i in range(len(g)):
                if g[i] == '?':
                    for val in domains[i]:
                        if val != instance[i]:
                            g_new = g[:i] + [val] + g[i+1:]
                            G_new.append(g_new)
        return G_new
    domains = [set(data[col]) for col in data.columns[:-1]]
    S = initialize_hypothesis(data)
    G = [initialize_hypothesis(data)]
    for index, row in data.iterrows():
        instance = row[:-1]
        label = row[-1]
        if label == 'Yes':
            S = generalize_S(S, instance)
            G = [g for g in G if is_more_general(g, S)]
        elif label == 'No':
            G = specialize_G(G, instance, domains)
            G = [g for g in G if is_more_general(S, g)]
    return S, G
S, G = candidate_elimination(data)
print("Most Specific Hypothesis:")
print(S)
print("\nGeneral Hypotheses:")
for g in G:
    print(g)
