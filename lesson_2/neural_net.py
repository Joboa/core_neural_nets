"""
Neural Network
"""

import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass: pass data through a model from beginning to end
    def forward(self, inputs):
        # Calculates outputs from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:
    def forward(self, inputs):

        # Unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 

        # Normalize each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create first dense layer with 2 input features and 3 outpt values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation for dense1
activation1 = Activation_ReLU()

# Create second layer with 3 input features and 3 output values
dense2 = Layer_Dense(3,3)

# Softmax activation for dense2
activation2 = Activation_Softmax()

# Perform forward pass of the training data
dense1.forward(X)

# Forward pass of dense1 through activation function (ReLU)
activation1.forward(dense1.output)

# Perform forward passs through the second dense layer
dense2.forward(activation1.output)

# Perform a forward pass through the activation function (Softmax)
activation2.forward(dense2.output)

print(activation2.output[:5])