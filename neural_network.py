import numpy as np

x = np.around(np.random.uniform(size=4), decimals=2)

weights = np.around(np.random.uniform(size=4), decimals= 2)

biases = np.around(np.random.uniform(size=1), decimals=2)

def weighted_sum(x, weights, biases):
    z =  np.sum(x * weights) + biases
    return z

def activation_function(weights_sum, activation_type):
    if activation_type == 'sigmoid':
        return 1 / (1 + np.exp(-weights_sum))
    elif activation_type == 'tanh':
        return (np.exp(weights_sum) -  np.exp(-weights_sum)) / (np.exp(weights_sum) +  np.exp(weights_sum))
    elif activation_type == 'relu':
        if weights_sum > 0:
            return weights_sum
        else:
            return 0

activation = ['sigmoid', 'tanh', 'relu']


for type in activation:
    z = weighted_sum(x, weights, biases)
    print(f'{type} - {activation_function(z, type)}')
    