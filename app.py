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

#SCALE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#ONE HOT ENCODING
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(Y.reshape(-1, 1))

#SPLIT DATA
x_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

layer_1 = Layer(neurons=3, n_inputs=4,  activation='relu')
layer_2 = Layer(neurons=4,  n_inputs=3, activation='relu')
layer_3 = Layer(neurons=4,  n_inputs=4, activation='relu')
layer_4 = Layer(neurons=3, n_inputs=4, activation='softmax')

nn = NeuralNetwork(x_inputs=x_train, y_train=y_train, epochs=100000)
nn.add_layer(layer_1)
nn.add_layer(layer_2)
nn.add_layer(layer_3)
nn.add_layer(layer_4)
nn.train()

for i in range(5):
    index_real_value = encoder.inverse_transform(y_test[i].reshape(1, -1))
    print(f'Expected value: {values[index_real_value[0][0]]}')
    accuracy, index = nn.predict(X_test[i].reshape(1, -1))
    print(f'{values[index]} with accuracy: {accuracy*100:.2f}%')
    print('---'*30)
    