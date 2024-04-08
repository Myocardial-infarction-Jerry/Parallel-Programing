import json
import numpy as np
import os

path=os.path.dirname(os.path.abspath(__file__))+'/'
with open(path+'output.json','r') as f:
    data=json.load(f)
    
array=np.array(data['array']).reshape(-1,1)

sum=np.sum(array)

if abs(sum-data['sum'])<1E-6:
    print("The sum is correct.")
else:
    print("The sum is incorrect.")