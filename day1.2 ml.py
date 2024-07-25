data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]
def find_s_algorithm(data):
    hypothesis = ['0'] * (len(data[0]) - 1)
    for example in data:
        if example[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':  
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:  
                    hypothesis[i] = '?'
                    
    return hypothesis
final_hypothesis = find_s_algorithm(data)
print("Final Hypothesis:", final_hypothesis)
