# Refila Dyah Ghizanda Wardoyo (21091397041)
# Kodingan Single Neuron dengan Input Layer Feature 10 dan Neuron 1

# Inisialisasi menggunakan Numpy
import numpy as np

# Inisialisasi variabel
# Input layer feature 10
inputs = [2, 7, 5, 1, 9, 3, 8, 4, 10, 6]

# Weights sama dengan panjang Input yaitu 10
# Jumlah Weights sesuai dengan jumlah Neuron yaitu 1
weights = [0.1, 0.9, 0.3, 0.8, -0.4, 0.10, 0.6, 0.2, 0.7, 0.5]

# Jumlah bias sama dengan jumlah Neuron yaitu 1
bias = 4

# Output
output = np.dot(weights,inputs) + bias
print(output)
