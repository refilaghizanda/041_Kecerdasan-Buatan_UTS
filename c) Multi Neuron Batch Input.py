# Refila Dyah Ghizanda Wardoyo (21091397041)
# Kodingan Multi Neuron Batch Input dengan Input Layer Feature 10 dan Neuron 5
# Per Batch nya 6 Input

# Inisialisasi menggunakan Numpy
import numpy as np

# Inisialisasi variabel
# Input layer feature 10
# Per Batch 6 Input
inputs = [[0.2, 1.7, 2.5, 3.1, 4.9, 5.3, 6.8, 1.4, 1.0, 6.1],
          [2.1, 3.2, 1.2, 4.2, 3.4, 2.6, 3.7, 9.1, 2.0, 3.4],
          [3.4, 5.9, 9.3, 7.7, 1.5, 4.8, 7.2, 9.9, 3.9, 1.6],
          [3.1, 4.4, 2.3, 4.5, 4.2, 3.2, 4.0, 1.0, 3.4, 2.9],
          [2.6, 3.7, 9.1, 2.0, 3.4, 0.2, 1.7, 2.5, 3.1, 4.1],
          [1.8, 3.7, 4.2, 2.5, 2.9, 9.1, 2.0, 3.4, 9.1, 2.1]]

# Weights sama dengan panjang Input yaitu 10
# Jumlah Weights sesuai dengan jumlah Neuron yaitu 5
weights = [[1.8, 3.7, 4.2, 2.5, 2.9, 1.1, 4.9, 5.3, 6.8, 1.4],
           [2.1, 5.3, 4.2, 2.3, 2.7, 3.7, 8.3, 4.9, 7.1, 6.6],
           [2.7, 2.5, 3.1, 4.1, 2.1, 3.2, 1.2, 5.1, 2.4, 9.0],
           [4.2, 3.2, 4.0, 1.0, 3.4, 2.9, 1.1, 3.1, 2.3, 2.2],
           [4.2, 5.9, 4.3, 8.7, 2.1, 5.3, 4.2, 2.3, 3.9, 8.1]]

biases = [3.1, 2.3, 3.4, 1.2, 3.6]

# Output
layer_outputs = np.dot(inputs,np.array(weights).T) + biases
print(layer_outputs)
