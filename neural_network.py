import numpy as np

x1 = 0.3 #Inputs
x2 = 0.6 #Inputs

weights = np.around(np.random.uniform(size=2), decimals=2)

biases = np.around(np.random.uniform(size=1), decimals=2)

z = (x1 * weights[0]) + (x2 * weights[1]) + biases[0] #Weighted sum

print(f'Weighted sum: {z}')

a = 1/ (1+ np.exp(-z)) #Activation Function

print(f'Activation Funtion Output: {a}')