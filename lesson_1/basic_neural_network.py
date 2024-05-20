"""
Neural networks

Output = input * weight + bias

Weight controls the signal between neurons. Thus, it decides how much influence
the input will have on the output.

Bias is a constant term added to the weighted sum before applying the activation 
function. It adjust the network to make accurate predictions.
"""

## Using pure python

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Output layer
layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight

    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)

print("pure:", layer_outputs)

## Using numpy and python

import numpy as np

outputs = np.dot(weights, inputs) + biases

# case with matrix as inputs
# outputs = np.dot(inputs, np.array(weights).T) + biases

print("new: ", outputs)