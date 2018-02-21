from __future__ import print_function

import numpy as np
np.random.seed(42)

# Define Sigmoid Activation and it's derivative
def activate(x):
    return 1 / (1+np.exp(-x))
def d_activation(x):
    return x * (1-x)

# Prepare the dataset
input_data =  np.array([[0,1,0,1,1,0],
                        [1,1,1,1,0,0],
                        [1,0,1,0,0,0],
                        [0,1,1,1,1,0],
                        [0,0,1,1,0,0],
                        [1,0,0,0,1,0],
                        [1,1,0,0,1,0],
                        [1,0,0,1,0,0],
                        [0,0,0,0,1,0],
                        [0,1,0,0,0,0],
                        [0,0,0,0,0,0],
                        [1,1,0,1,1,0],
                        [1,0,1,0,1,0],
                        [0,0,1,0,1,0],
                        [1,0,1,1,0,0],
                        [0,1,1,0,0,0],
                        [0,1,0,0,1,0],
                        [0,1,0,1,0,0],
                        [1,1,1,1,1,0],
                        [1,0,0,1,1,0],
                        [1,1,1,0,1,0],
                        [0,1,1,1,0,0],
                        [1,1,0,1,0,0],
                        [0,0,1,1,1,0],
                        [0,0,0,1,1,0],
                        [0,0,1,0,0,0],
                        [1,0,1,1,1,0],
                        [0,1,1,0,1,0],
                        [1,1,1,0,0,0],
                        [1,1,0,0,0,0],
                        [0,0,0,1,0,0],
                        [1,0,0,0,0,0]])
labels =  np.array([[1],
                    [0],
                    [0],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [1],
                    [0],
                    [0],
                    [1],
                    [0],
                    [1],
                    [1],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0],
                    [0],
                    [0],
                    [0]])

test_data  =  np.array([[1,0,1,1,0,1],
                        [1,1,1,1,0,1],
                        [0,1,0,0,1,1],
                        [1,1,0,1,1,1],
                        [0,0,0,0,0,1],
                        [0,1,1,1,0,1],
                        [0,1,1,0,0,1],
                        [1,1,1,0,0,1],
                        [0,1,1,0,1,1],
                        [0,1,0,1,0,1],
                        [0,0,0,1,1,1],
                        [1,0,1,0,0,1],
                        [0,0,0,0,1,1],
                        [1,1,0,0,0,1],
                        [0,1,1,1,1,1],
                        [1,1,0,0,1,1],
                        [1,0,1,0,1,1],
                        [0,1,0,0,0,1],
                        [0,0,1,0,0,1],
                        [1,0,0,1,1,1],
                        [1,1,0,1,0,1],
                        [0,1,0,1,1,1],
                        [0,0,1,1,0,1],
                        [1,1,1,0,1,1],
                        [1,0,0,1,0,1],
                        [0,0,0,1,0,1],
                        [1,0,0,0,1,1],
                        [0,0,1,1,1,1],
                        [1,1,1,1,1,1],
                        [0,0,1,0,1,1],
                        [1,0,1,1,1,1],
                        [1,0,0,0,0,1]])

# Training Parameters                        
num_epochs = 60000
n_samples = input_data.shape[0]
batch_size = 16

# Weights and Biases
weight_i_h = np.random.random((6,16))
weight_h_o = np.random.random((16,1))
bias_i_h = np.random.random((1, 16))
bias_h_o = np.random.random((1, 1))

# Lists to store epoch and error
x = []
y = []

# Train the model
for epoch in range(1,num_epochs+1):
    
    for batch in range(n_samples/batch_size):
        
        # Forward Propagation
        input_layer = input_data[batch_size*batch: batch_size*(batch+1)]
        hidden_layer = activate(np.dot(input_layer, weight_i_h) + bias_i_h)
        output_layer = activate(np.dot(hidden_layer, weight_h_o) + bias_h_o)
        output_labels = labels[batch_size*batch: batch_size*(batch+1)]

        loss = output_layer - output_labels
        
        # Backward Propagation
        weight_i_h -= 0.01 * input_layer.T.dot(((loss * d_activation(output_layer)).dot(weight_h_o.T)) * d_activation(hidden_layer))
        bias_i_h -= 0.01 * sum(((loss * d_activation(output_layer)).dot(weight_h_o.T)) * d_activation(hidden_layer))
        weight_h_o -= 0.01 * hidden_layer.T.dot(loss * d_activation(output_layer))
        bias_h_o -= 0.01 * sum(loss * d_activation(output_layer))
    
    # Store checkpoints for plot
    if epoch == 1 or epoch % 6000 == 0:
        print ('Epoch: ', '%06d' % (epoch), 'Loss: {0:.5f}'.format((sum(abs(loss)))[0]))
        x.append(epoch)
        y.append((sum(abs(loss)))[0])

# Plot the Loss
import matplotlib.pyplot as plt
plt.plot(x[1:], y[1:], label = 'Loss')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Test the model
test = test_data[:16]
prediction = activate((activate(test.dot(weight_i_h) + bias_i_h)).dot(weight_h_o) + bias_h_o)
print ((np.round(prediction, decimals=0)).astype(int))
