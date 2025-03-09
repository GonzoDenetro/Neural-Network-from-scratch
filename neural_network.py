import numpy as np

np.random.seed(42)


class NeuralNetwork():
    def __init__(self, n_neurons, n_layers, x_inputs):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.weights = []
        self.biases = []
        self.network = {}
        self.x_inputs = x_inputs
    
    def __str__(self):
        text = ""
        text = f'Input layer: {len(self.x_inputs)} neurons. '
        for i in range(len(self.n_neurons)):
            if i == self.n_layers-1:
                text += f'Output layer: {self.n_neurons[i]} neurons. '
            else:
                text += f'Hidden layer {i+1}: {self.n_neurons[i]} neurons. '
        return text
                
    
    
    #SUM OF WEIGHTS
    def weighted_sum(self, x, weights, biases):
        z =  x.dot(weights.T) + biases
        return z

    #ACTIVATION FUNCTION
    def activation_function(self, weights_sum, activation_type):
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
    def neuron_outputs(self, inputs, weights, biases):
        z = self.weighted_sum(inputs, weights, biases)
        return self.activation_function(z, 'sigmoid')

    #This function creates the parameters that the layer will need, the parameters depends on the number of neurons and the inputs we have
    def layer_ouput(self, neurons, inputs):
        #print(f'inputs shape: {inputs.shape}')
        #print(f'weights shape: {neurons}, {inputs.shape[0]}')
        weights = np.random.rand(neurons, inputs.shape[0])
        biases = np.random.rand(neurons)
        output = self.neuron_outputs(inputs, weights, biases)
        return [weights, biases, output]
    
    def forward_propagation(self, inputs):
        for i in range(self.n_layers):
            layer = {}
            layer_name = f'layer-{i}'
            layer['name'] = layer_name
            
            if i == 0:
                l_ouput = self.layer_ouput(self.n_neurons[i], inputs)
            else:
                l_ouput = self.layer_ouput(self.n_neurons[i], result) #The output of the past layer is the input for the next layer
            
            layer['weights'] = l_ouput[0]
            self.weights = l_ouput[0]
            
            layer['biases'] = l_ouput[1]
            self.biases = l_ouput[1]
            
            result = l_ouput[2]
            layer[f'output-a{i}'] = result
            self.network[layer_name] = layer

        
x = np.around(np.random.uniform(size=2), decimals=2)
       
        
n_network = NeuralNetwork(n_neurons=[3, 3, 2], n_layers=3, x_inputs=x)
n_network.forward_propagation(inputs=x)
print(n_network)
print(n_network.network)

    