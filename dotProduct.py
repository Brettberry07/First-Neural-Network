import numpy as np

input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17,0.32]

# What the dot product is
"""
The dot product is a simple way of seeing how similar two vectors are.
Depending on the number we get back is how similar they are.
With the dot product of weights_1 and input_vector only being 2.1672,
while the dot product of weights_2 and input_vector is 4.1259; we know
that weights_2 and input_vector is more similar than weights_1 and input_vecotr.
"""

# Computing the dot product of input_vector and weights_1 manually
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult

print(f"The dot product is: {dot_product_1}")
# Output: 2.1672

# Computing the dot product of input_vector and weights_2 in numpy
dot_product_2 = np.dot(input_vector,weights_2)

print(f"The dot product is: {dot_product_2}")
# Output: 4.1259

