import numpy as np

x = np.around(np.random.uniform(size=2), decimals=2)

weights = np.around(np.random.uniform(size=4), decimals= 2)

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

    
def neuron(inputs, weights, biases):
    z = weighted_sum(inputs, weights, biases)
    return activation_function(z, 'sigmoid')

def layer_ouput(n_neurons, inputs):
    weights = np.random.rand(n_neurons, inputs.shape[0])
    biases = np.random.rand(n_neurons)
    output = neuron(inputs, weights, biases)
    return [weights, biases, output]
    
def network(n_layers, inputs, n_neurons):
    my_network = {}
    
    for i in range(n_layers):
        layer = {}
        layer_name = f'layer-{i}'
        layer['name'] = layer_name
        if i == 0:
            l_ouput = layer_ouput(n_neurons[i], inputs)
            l_weights = l_ouput[0]
            layer['weights'] = l_weights
            l_biases = l_ouput[1]
            layer['biases'] = l_biases
            layer_output = l_ouput[2]
            layer['output'] = layer_output
        else:
            l_ouput = layer_ouput(n_neurons[i], layer_output)
            l_weights = l_ouput[0]
            layer['weights'] = l_weights
            l_biases = l_ouput[1]
            layer['biases'] = l_biases
            layer_output = l_ouput[2]
            layer['output'] = layer_output
        my_network[layer_name] = layer 
        return my_network       
            
        

print(''*20)
print(network(3, x, [3, 3, 2]))
    
    

    