import json
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__)) + '/'
with open(path+'output.json', 'r') as f:
    data = json.load(f)

A = np.array(data['A']).reshape(-1, data['n'])
B = np.array(data['B']).reshape(data['n'], -1)
C = np.array(data['C']).reshape(-1, data['k'])

AB = np.matmul(A, B)

if np.allclose(AB, C):
    print("The matrix multiplication is correct.")
else:
    print("The matrix multiplication is incorrect.")
