import json
import numpy as np

# Read the values from output.json
with open('output.json', 'r') as file:
    data = json.load(file)
    n = data['n']
    A = np.array(data['A']).reshape(n, n)
    B = np.array(data['B']).reshape(n, n)
    C = np.array(data['C']).reshape(n, n)

# Perform matrix multiplication
C_ = np.dot(A, B)
is_correct = np.allclose(C, C_)

if is_correct:
    print("The result is correct!")
else:
    print("The result is incorrect!")
