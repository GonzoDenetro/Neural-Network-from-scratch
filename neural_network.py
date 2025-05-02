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
                exps = np.exp(weights_sum - np.max(weights_sum, axis=1, keepdims=True))  # estabilidad numérica
                return exps / np.sum(exps, axis=1, keepdims=True)
        except Exception as e:
            print(f'Error: {e}')
            print('No activation Function Found')
            print(f'Activation: {self.activation}')
    
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
    def __init__(self, x_inputs, y_train,  epochs=1000):
        self._layers = []
        self._output = []
        self.x_inputs = x_inputs 
        self.y_train = np.array(y_train)
        self.epochs = epochs
    
    def add_layer(self, layer):
        self._layers.append(layer)
    
    def layers(self):
        return self._layers    
    
    def propagate_network(self, inputs):
        for i, layer in enumerate(self._layers):
            if i == 0:
                layer.forward_propagation(x_inputs=inputs)
            else:
                layer.forward_propagation(x_inputs=self._layers[i-1].values())
            
    def output(self):
        last_layer = len(self._layers) - 1
        self._output = self._layers[last_layer].values()
        return self._output
    
    def loss_function(self):
        cross_entroy = -1 * np.sum(self.y_train * np.log(self._output + 1e-9), axis=1) 
        cross_entroy = np.mean(cross_entroy)
        return cross_entroy
    
    def cross_entropy_derivate(self):
        return - self.y_train / self._output
    
    def backpropgation(self):
        last_layer = len(self._layers) - 1
        output = self.output()
        learning_rate = 0.001
        delta = self.cross_entropy_derivate()
        
        for i in range(last_layer, -1, -1):
            #print(f'--------------LAYER {i}-----------------')
            
            if self._layers[i].activation == 'relu':
                values = self._layers[i].values()
                de_activation = (values > 0).astype(float)
                return de_activation
            elif self._layers[i].activation == 'softmax':
                de_activation = self._layers[i].softmax_derivate(output)
            
            input_to_layer = self.x_inputs if i == 0 else self._layers[i-1].values()
            
            delta = delta * de_activation
            delta_weights = np.dot(delta.T, input_to_layer)
            
            #print(f'Weights: {self._layers[i].weights}')
            
            #Gradient Descent
            self._layers[i].weights = self._layers[i].weights - (learning_rate*delta_weights)
            self._layers[i].biases = self._layers[i].biases - (learning_rate *  np.sum(delta, axis=0))
            
            #print(f'Weights New: {self._layers[i].weights}')
            
            #Backpropagation (next layer)
            #delta = Wᵗ ⋅ δ
            delta = np.dot(delta, self._layers[i].weights)
            
    def train(self):
        for i in range(self.epochs):
            self.propagate_network(self.x_inputs)
            self.backpropgation()
            
            if i % 100 == 0:
                print(f'Loss: {self.loss_function()}')
    
    def predict(self, x_test):
        self.propagate_network(x_test)
        return self.output()
                            
            