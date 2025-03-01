import numpy as np

x = np.around(np.random.uniform(size=2), decimals=2)


#SUM OF WEIGHTS
def weighted_sum(x, weights, biases):
    z =  np.sum(x.dot(weights.T)) + biases
    return z

#ACTIVATION FUNCTION
def activation_function(weights_sum, activation_type):
    if activation_type == 'sigmoid':
        return 1 / (1 + np.exp(-weights_sum))
    elif activation_type == 'tanh':
        return (np.exp(weights_sum) -  np.exp(-weights_sum)) / (np.exp(weights_sum) +  np.exp(-weights_sum))
    elif activation_type == 'relu':
        if weights_sum > 0:
            return weights_sum
        else:
            return 0

#This function represents the operation that a neuron performs    
def neuron(inputs, weights, biases):
    z = weighted_sum(inputs, weights, biases)
    return activation_function(z, 'sigmoid')


#This function creates the parameters that the layer will need, the parameters depends on the number of neurons and the inputs we have
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
        else:
            l_ouput = layer_ouput(n_neurons[i], result) #The output of the past layer is the input for the next layer
        
        layer['weights'] = l_ouput[0]
        layer['biases'] = l_ouput[1]
        result = l_ouput[2]
        layer[f'output-a{i}'] = result
        my_network[layer_name] = layer
        
    return my_network       
            
        
n_network = network(n_layers=3, inputs=x, n_neurons=[3, 3, 2])
print(n_network['layer-0'])

    