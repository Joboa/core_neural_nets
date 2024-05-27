"""
Soft max activation function: For classification
y = e^z(i,j)/[n=1 to N]sum(e^z(i,n))
"""

import numpy as np
import math

layer_outputs = [4.8, 1.21, 2.385]

# Method 1
E = math.e

# exp_values = []

# Exponentiation of values
# for output in layer_outputs:
#     exp_values.append(E**output)

# # Normalization of values
# norm_base = sum(exp_values)
# norm_values = []

# for value in exp_values:
#     norm_values.append(value / norm_base)

# print('Normalized exponentiated values:')
# print(norm_values)
# print('Sum of normalized values:', sum(norm_values))


# Method 2: Using numpy

# exp_values = np.exp(layer_outputs)
# print('exponentiated values:')
# print(exp_values)

# # Normalize values
# norm_values = exp_values / np.sum(exp_values)
# print('normalized exponentiated values:')
# print(norm_values)
# print('sum of normalized values:', np.sum(norm_values))


# Method 3: class-based
# Softmax activation

class Activation_Softmax:
    def forward(self, inputs):

        # Unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Numerical stability

        """
        Numerical stability
        Is preventing numerical overflow or underflow, resulting in NaNs or infinites 
        during computation.
        
        Suppose x = [1001, 1002, 1003]
        e^x result in very large numbers, potentially causing overflow

        Hence "inputs - np.max(inputs)" where max = 1003
        Thus = [-2, -1, 0]. 

        The above results in a manageable numbers that are less likely to cause overflow
        """

        # Normalize each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Testing numerical stability
softmax = Activation_Softmax()
softmax.forward([[1,2,3]])
print("1: ", softmax.output)

softmax.forward([[-2,-1,0]])
print("2: ", softmax.output)