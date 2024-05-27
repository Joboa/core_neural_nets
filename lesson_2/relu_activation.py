"""
Rectified Linear Unit (ReLU) activation funtion

y = {
     x > 0, x
     x <= 0, x
}
"""

import numpy as np
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# Method 1
output = []
# for i in inputs:
#     if i > 0:
#         output.append(i)
#     else:
#         output.append(0)

# print(output)

# Method 2
# for i in inputs:
#     output.append(max(0, i))

# print(output)

# Method 3
# output = np.maximum(0, inputs)
# print(output)

# Method 4: class representation


class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
