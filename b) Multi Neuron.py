# Refila Dyah Ghizanda Wardoyo (21091397041)
# Kodingan Multi Neuron dengan Input Layer Feature 10 dan Neuron 5

# Inisialisasi menggunakan Numpy
import numpy as np

# Inisialisasi variabel
# Input layer feature 10
inputs = [0.2, 1.7, 2.5, 3.1, 4.9, 5.3, 6.8, 1.4, 1.0, 6.1]

# Weights sama dengan panjang Input yaitu 10
# Jumlah Weights sesuai dengan jumlah Neuron yaitu 5
weights = [[0.3, 0.5, 0.9, 0.8, 0.6, 0.2, 0.1, 0.2, 0.4, -0.4],
           [0.12, -0.21, 0.24, 0.2, 0.18, -0.23, 0.4, 0.77, 0.29, 0.39],
           [0.14, 02.6, 0.7, -0.3, 0.34, 0.29, -0.56, -0.78, 0.19, -0.1],
           [0.1, 0.9, 0.3, 0.8, -0.4, 0.10, 0.6, 0.2, 0.7, 0.5],
           [0.15, 0.17, -0.10, 0.16, 0.11, -0.9, 0.19, 0.13, 0.18, 0.4]]

# Jumlah bias sama dengan jumlah Neuron yaitu 5
biases = [8.1, 2.7, 3.1, 1.4, 5.6]

# Output
layer_outputs = np.dot(weights,inputs) + biases
print(layer_outputs)
