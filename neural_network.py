import numpy as np

x = np.around(np.random.uniform(size=4), decimals=2)

weights = np.around(np.random.uniform(size=4), decimals= 2)

biases = np.around(np.random.uniform(size=1), decimals=2)

def weighted_sum(x, weights, biases):
    z =  np.sum(x * weights) + biases
    return z

print(weighted_sum(x, weights, biases))
