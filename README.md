# Neural Network From Scratch 

This project implements a neural network entirely from scratch using NumPy, aimed at deeply understanding how artificial neural networks work without using high-level libraries like TensorFlow or PyTorch. The Iris dataset is used for training and validation.

## Project Objective

The main goal is to learn the inner workings of artificial neural networks, by manually implementing each component: from activation functions and forward propagation to backpropagation and gradient calculation.



## Neural Network Architecture

The neural network is composed of multiple layers (`Layer`), each with:
- Neurons
- Activation function (`relu`, `sigmoid`, `tanh`, `softmax`)
- Weights and biases initialized randomly

## Features

- Manual **forward propagation** implementation
- Manual **backpropagation** implementation
- Custom **loss function (cross entropy)**
- **Gradient descent** for training
- **Model accuracy evaluation**

## Dataset

The Iris Dataset contains 150 flower samples classified into 3 species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

Each sample has 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

