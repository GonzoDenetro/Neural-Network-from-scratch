import numpy as np

np.random.seed(42)


class Layer():
    def __init__(self, neurons, n_inputs, activation):
        self.neurons = neurons
        self.x_inputs = None
        self.activation = activation
        self.weights = np.random.rand(neurons, n_inputs)
        self.biases = np.random.rand(neurons)
        self._values = np.zeros(neurons)
        
    def weighted_sum(self, x_inputs):
        return x_inputs.dot(self.weights.T) + self.biases
    
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
    
    def forward_propagation(self, x_inputs):
        self.x_inputs = x_inputs
        weights_sum = self.weighted_sum(self.x_inputs)
        self._values = self.activation_function(weights_sum)

    def values(self):
        return self._values
    
    def softmax_derivate(self, output):
        return output * (1 - output)

    def weights_sum_derivate(self):
        print(f'jjfjfjf {self.x_inputs}')
        return self.x_inputs.T
        


class NeuralNetwork():
    def __init__(self, x_inputs, y_train):
        self._layers = []
        self._output = []
        self.x_inputs = x_inputs 
        self.y_train = np.array(y_train)
    
    def add_layer(self, layer):
        self._layers.append(layer)
    
    def layers(self):
        return self._layers    
    
    def propagate_network(self):
        for i, layer in enumerate(self._layers):
            if i == 0:
                layer.forward_propagation(x_inputs=self.x_inputs)
            else:
                layer.forward_propagation(x_inputs=self._layers[i-1].values())
            
    def output(self):
        last_layer = len(self._layers) - 1
        self._output = self._layers[last_layer].values()
        return self._output
    
    def loss_function(self):
        cross_entroy = -1 * np.sum(self.y_train * np.log(self._output))
        return cross_entroy
    
    def cross_entropy_derivate(self):
        return - self.y_train / self._output
    
    def backpropgation(self):
        last_layer = len(self._layers) - 1
        output = self.output()
        learning_rate = 0.001
        delta = self.cross_entropy_derivate()
        
        for i in range(last_layer, -1, -1):
            print(f'--------------LAYER {i}-----------------')
            
            if self._layers[i].activation == 'relu':
                de_activation = 1
            elif self._layers[i].activation == 'softmax':
                de_activation = self._layers[i].softmax_derivate(output)
            
            input_to_layer = self.x_inputs if i == 0 else self._layers[i-1].values()
            
            delta = delta * de_activation
            delta_weights = np.outer(delta, input_to_layer)
            
            print(f'Weights: {self._layers[i].weights}')
            
            #Gradient Descent
            self._layers[i].weights = self._layers[i].weights - (learning_rate*delta_weights)
            self._layers[i].biases -= learning_rate * delta
            
            print(f'Weights New: {self._layers[i].weights}')
            
            #Backpropagation (next layer)
            delta =  self._layers[i].weights.T.dot(delta)
            
                            
            
            

x = np.around(np.random.uniform(size=2), decimals=2)

layer_1 = Layer(neurons=3, n_inputs=2,  activation='relu')
layer_2 = Layer(neurons=3,  n_inputs=3, activation='relu')
layer_3 = Layer(neurons=2, n_inputs=3, activation='softmax')

network = NeuralNetwork(x_inputs=x, y_train=[0, 1])
network.add_layer(layer_1)
network.add_layer(layer_2)
network.add_layer(layer_3)
network.propagate_network()



print(f'FINAL OUTPUT: {network.output()}')
print(f'Loss: {network.loss_function()}')
network.backpropgation()
#print(f'Tetha: {network.backpropgation()}')

#print(f'Layer 1 OUTPUT {layer_1._values}') 