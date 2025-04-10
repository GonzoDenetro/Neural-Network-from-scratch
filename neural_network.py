import numpy as np

np.random.seed(42)


class Layer():
    def __init__(self, neurons, activation, x_inputs):
        self.neurons = neurons
        self.activation = activation
        self.x_inputs = x_inputs
        self.weights = np.random.rand(neurons, x_inputs.shape[0])
        self.biases = np.random.rand(neurons)
        self._values = np.zeros(neurons)
        
    def weighted_sum(self):
        return self.x_inputs.dot(self.weights.T) + self.biases
    
    def activation_function(self, weights_sum):
        try:
            if self.activation == 'sigmoid':
                return 1 / (1 + np.exp(-weights_sum))
            elif self.activation == 'tanh':
                return (np.exp(weights_sum) -  np.exp(-weights_sum)) / (np.exp(weights_sum) +  np.exp(-weights_sum))
            elif self.activation == 'relu':
                return np.maximum(0, weights_sum)
            elif self.activation == 'softmax':
                return np.exp(weights_sum) / np.sum(np.exp(weights_sum))
        except:
            print('No activation Function Found')
    
    def forward_propagation(self):
        weights_sum = self.weighted_sum()
        self._values = self.activation_function(weights_sum)

    def values(self):
        return self._values


class NeuralNetwork():
    def __init__(self):
        self._layers = []
        self._output = []
    
    def add_layer(self, layer):
        self._layers.append(layer)
    
    def layers(self):
        return self._layers    
    
    def propagate_network(self):
        for layer in self._layers:
            layer.forward_propagation()
            
    def output(self):
        last_layer = len(self._layers) - 1
        self._output = self._layers[last_layer].values()
        return self._output
    
    def loss_function(self, y_values):
        cross_entroy = -1 * np.sum(y_values * np.log(self._output))
        return cross_entroy

x = np.around(np.random.uniform(size=2), decimals=2)
print(x)

layer_1 = Layer(3, 'relu', x)
layer_2 = Layer(3, 'relu', layer_1.values())
layer_3 = Layer(2, 'softmax', layer_2.values())

network = NeuralNetwork()
network.add_layer(layer_1)
network.add_layer(layer_2)
network.add_layer(layer_3)
network.propagate_network()


print(network.output())
print(network.loss_function([0, 1]))
        
    