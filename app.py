import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork, Layer

#LOAD DATA 
data = load_iris()
X = data.data
Y = data.target #  (0 = setosa, 1 = versicolor, 2 = virginica)
values = data.target_names
#print(values)
#print(Y)

#SCALE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(X_scaled)

#ONE HOT ENCODING
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(Y.reshape(-1, 1))
#print(y_onehot)

#SPLIT DATA
x_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)
print(f'X_train: {x_train.shape}')


layer_1 = Layer(neurons=3, n_inputs=4,  activation='relu')
layer_2 = Layer(neurons=4,  n_inputs=3, activation='relu')
layer_3 = Layer(neurons=4,  n_inputs=4, activation='relu')
layer_4 = Layer(neurons=3, n_inputs=4, activation='softmax')

nn = NeuralNetwork(x_inputs=x_train, y_train=y_train, epochs=10000)
nn.add_layer(layer_1)
nn.add_layer(layer_2)
nn.add_layer(layer_3)
nn.add_layer(layer_4)
nn.train()

print(y_test[0])


print(nn.predict(X_test[0].reshape(1, -1)))

#network = NeuralNetwork(x_inputs=x, y_train=[0, 1])
#network.add_layer(layer_1)
#network.add_layer(layer_2)
#network.add_layer(layer_3)
#network.propagate_network()



#print(f'FINAL OUTPUT: {network.output()}')
"""print(f'Loss: {network.loss_function()}')
network.backpropgation()
network.propagate_network()
print(f'Loss: {network.loss_function()}')
"""
#network.train()
#print(f'Tetha: {network.backpropgation()}')

#print(f'Layer 1 OUTPUT {layer_1._values}') 
