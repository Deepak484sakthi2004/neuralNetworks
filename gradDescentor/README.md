# gradDescentor - Neural Network from Scratch

This project implements a simple neural network from scratch without relying on external deep learning libraries such as PyTorch or TensorFlow. The neural network is defined within a Python class named NeuralNetwork, allowing users to understand the fundamental concepts of building and training neural networks.

## Features

- Custom Implementation: No external deep learning libraries are used; the neural network is coded from scratch.
- Backpropagation: The project demonstrates the backpropagation algorithm for training the neural network.
- Gradient Descent: The neural network utilizes gradient descent to optimize weights during training.
- Predictions: Once trained, the model can make predictions on new data by forward-passing through the network.

## Getting Started

- Object Creation: Instantiate the Value class to create an object capable of performing various mathematical operations.

- Basic Operations:
  - add(x): Addition operation.
  - multiply(x): Multiplication operation.
  - subtract(x): Subtraction operation.
  - exponent(x): Exponential operation.
  - tanh(x)    : Sigmoid operation.

- Gradient Chain Rule:
  - The Value object incorporates the chain rule of derivatives to compute gradients efficiently.

- Visualize Neural Network:
  - Utilize the draw_dot function to visualize the neural network structure .

- Backpropagation:
  - The Value class includes a backpropagation function for training the neural network.
