# Refila Dyah Ghizanda Wardoyo (21091397041)
# Kodingan Multi Neuron Batch Input dengan Input Layer Feature 10 
# Per batch nya 6 Input
# Hidden Layer 1 yaitu 5 Neuron
# Hidden Layer 2 yaitu 3 Neuron

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

# Weights 1 sama dengan panjang Input yaitu 10
# Jumlah Weights 1 sesuai dengan jumlah Neuron yaitu 5
weights1 = [[1.8, 3.7, 4.2, 2.5, 2.9, 1.1, 4.9, 5.3, 6.8, 1.4],
           [2.1, 5.3, 4.2, 2.3, 2.7, 3.7, 8.3, 4.9, 7.1, 6.6],
           [2.7, 2.5, 3.1, 4.1, 2.1, 3.2, 1.2, 5.1, 2.4, 9.0],
           [4.2, 3.2, 4.0, 1.0, 3.4, 2.9, 1.1, 3.1, 2.3, 2.2],
           [4.2, 5.9, 4.3, 8.7, 2.1, 5.3, 4.2, 2.3, 3.9, 8.1]]

# Inisialisasi Biases Layer 1 sama dengan Neuron yaitu 5
biases1 = [3.1, 2.3, 3.4, 1.2, 3.6]

# Weights 2 sama dengan neuron layer 1 yaitu 5
# Jumlah Weights 2 sesuai dengan neuron layer 2 yaitu 3 neuron
weights2 = [[0.4, 1.3, 1.3, 2.4, -1.1],
			[0.5, 1.2, 3.1, 3.0, 2.1],
			[2.3, 1.5, 2.0, 3.1, 1.4]]

# Inisialisasi Biases Layer 1 sama dengan Neuron yaitu 3
biases2 =  [1.0, 1.2, 2.4]

# Output
# Menghitung Layer 1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# Output
# Menghitung Layer 2 dengan hasil perhitungan di Layer 1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs) 