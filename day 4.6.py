import pandas as pd
data = pd.DataFrame({
    'Size': ['Big', 'Small', 'Small', 'Big', 'Small'],
    'Color': ['Red', 'Red', 'Red', 'Blue', 'Blue'],
    'Shape': ['Circle', 'Triangle', 'Circle', 'Circle', 'Circle'],
    'Class': ['No', 'No', 'Yes', 'No', 'Yes']
})
def initialize_specific_hypothesis(data):
    return ['0'] * (len(data.columns) - 1)
def candidate_elimination(data):
    specific_hypothesis = initialize_specific_hypothesis(data)
    general_hypotheses = [initialize_specific_hypothesis(data)]
    for _, row in data.iterrows():
        if row['Class'] == 'Yes':
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i] == '0':
                    specific_hypothesis[i] = row[i]
                elif specific_hypothesis[i] != row[i]:
                    specific_hypothesis[i] = '?'
            general_hypotheses = [g for g in general_hypotheses if not is_more_general(g, specific_hypothesis)]
        elif row['Class'] == 'No':
            general_hypotheses = [g for g in general_hypotheses if not is_more_general(specific_hypothesis, g)]
            general_hypotheses.append(specific_hypothesis.copy())
            specific_hypothesis = ['0'] * (len(data.columns) - 1)
    return specific_hypothesis, general_hypotheses
def is_more_general(h1, h2):
    for i in range(len(h1)):
        if h1[i] != '?' and (h1[i] != h2[i] and h2[i] != '?'):
            return False
    return True
specific_hypothesis, general_hypotheses = candidate_elimination(data)
print("Most Specific Hypothesis:")
print(specific_hypothesis)
print("\nGeneral Hypotheses:")
for h in general_hypotheses:
    print(h)
