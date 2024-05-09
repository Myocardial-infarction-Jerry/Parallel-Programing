import json
import numpy as np

# Read the values from output.json
with open('output.json', 'r') as file:
    data = json.load(file)
    A = np.array(data['A'])
    B = np.array(data['B'])
    C = np.array(data['C'])
    n = data['n']

# Verify A + B = C
is_correct = np.allclose(A + B, C)

if is_correct:
    print("The result is correct!")
else:
    print("The result is incorrect!")
