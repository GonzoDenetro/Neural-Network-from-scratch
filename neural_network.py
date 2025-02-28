import numpy as np

x = np.around(np.random.uniform(size=2), decimals=2)

weights = np.around(np.random.uniform(size=4), decimals= 2)
print(x.shape[0])

biases = np.around(np.random.uniform(size=1), decimals=2)

#SUM OF WEIGHTS
def weighted_sum(x, weights, biases):
    z =  np.sum(x * weights) + biases
    return z

#ACTIVATION FUNCTION
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

def initialize(n_layers, n_neurons, x_inputs):
    network = {}
    network['weights'] = np.random.rand(n_neurons, x_inputs.shape[0])
    network['biases'] = np.random.rand(n_neurons)
    
    network['layers'] = n_layers
    print(network)
    

create_layer(1, 3, x)
    
    

    