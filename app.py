################################### training without saving ,,,also plot a loss###########

import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Preprocessing: Flatten images, normalize, and one-hot encode labels
# Normalize to range [0, 1]
x_train = x_train.reshape(len(x_train), 28 * 28) / 255.0  
x_test = x_test.reshape(len(x_test), 28 * 28) / 255.0


# One-hot encode labels
num_classes = 10
# print(len(y_train))
# print(len(y_test))
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]



# Define MLP Architecture
input_size = 28 * 28
hidden_size = 128
output_size = 10

# initialize weights and biases :دلی الدین منی

# from input to hidden
W1 = np.random.randn(input_size, hidden_size) * 0.01  
# This creates a 2D array with 1 row and hidden_size columns, where all elements are initialized to 0.
# When you perform a forward pass in a neural network, the pre-activation value for a hidden layer is typically computed as:Z=XW+b ==> X is the input matrix and  W is the weight matrix and b is the bias vector.
b1 = np.zeros((1, hidden_size))
# from hidden to output
W2 = np.random.randn(hidden_size, output_size) * 0.01  
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    # It outputs x if x>0, otherwise it outputs 0 0. like : x = np.array([-3, -1, 0, 1, 3]) output = relu(x) ==>  print(output)  # Output: [0 0 0 1 3]
    return np.maximum(0, x)

# The function you provided calculates the derivative of the ReLU (Rectified Linear Unit) activation function, which is used during backpropagation in a neural network to compute gradients. Here's an explanation of what it does:
def relu_derivative(x):
    #  x > 0 ==> This creates a boolean array where each element is True if the corresponding value in x is greater than 0, and False otherwise.
    #  astype(float) ==> Converts the boolean values to floats: True becomes 1.0, and False becomes 0.0.
    # This function computes the gradient (derivative) of the ReLU activation function with respect to its input, which is essential during backpropagation to update weights in the neural network.

    return (x > 0).astype(float)

def softmax(x):
    # Stability improvement
    # i got this function from google
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss
# find loss ...this is loss function
def cross_entropy_loss(y_pred, y_true):
    n = y_pred.shape[0]
    # Add epsilon for stability
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / n  
    return loss

# Training parameters
learning_rate = 0.005
# epochs = 50 # accuracy 98.13 %
epochs = 10 # accuracy 97.76 %


loss_history = []
# Training loop
for epoch in range(epochs):
    # Keep track of loss for the epoch
    total_loss = 0  

    for i in range(x_train.shape[0]):  # Process each training example sequentially
        x_sample = x_train[i:i + 1]  # Single sample
        y_sample = y_train_onehot[i:i + 1]  # Corresponding label

        # Forward pass
        z1 = np.dot(x_sample, W1) + b1  # Input to hidden
        a1 = relu(z1)                 # Activation
        z2 = np.dot(a1, W2) + b2      # Hidden to output
        a2 = softmax(z2)              # Activation

        # Compute loss for the sample
        loss = cross_entropy_loss(a2, y_sample)
        total_loss += loss  # Accumulate loss for the epoch

        # Backward pass ==>"i got this backward pass from chatGpt"
        dz2 = a2 - y_sample            # Gradient of loss w.r.t z2
        dW2 = np.dot(a1.T, dz2)       # Gradient of loss w.r.t W2
        db2 = np.sum(dz2, axis=0, keepdims=True)  # Gradient of loss w.r.t b2

        dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)  # Gradient of loss w.r.t z1
        dW1 = np.dot(x_sample.T, dz1)       # Gradient of loss w.r.t W1
        db1 = np.sum(dz1, axis=0, keepdims=True)  # Gradient of loss w.r.t b1

        # Update weights and biases using the gradients
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Save average loss for the epoch
    loss_history.append(total_loss )    

    # Compute accuracy on training data
    z1 = np.dot(x_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    predictions = np.argmax(a2, axis=1)
    accuracy = np.mean(predictions == y_train)

    # Print loss and accuracy
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")


# Test the model
z1 = np.dot(x_test, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = softmax(z2)
test_predictions = np.argmax(a2, axis=1)
test_accuracy = np.mean(test_predictions == y_test)

# Print test accuracy
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# plotting losses:
# Plot the loss history
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='b', label='Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.xticks(range(1, epochs + 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()






######################## if i saved trained weighted with .npz file,,i just use this code,,if not use above code:##############
# s
################################### if use Tensorflow ##############################3
# accuracy:92%

# # Importing necessary modules
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense

# # Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # Normalize image pixel values by dividing by 255 (grayscale)
# gray_scale = 255

# x_train = x_train.astype('float32') / gray_scale
# x_test = x_test.astype('float32') / gray_scale

# # Checking the shape of feature and target matrices
# print("Feature matrix (x_train):", x_train.shape)
# print("Target matrix (y_train):", y_train.shape)
# print("Feature matrix (x_test):", x_test.shape)
# print("Target matrix (y_test):", y_test.shape)

# # # Visualizing 100 images from the training data
# # fig, ax = plt.subplots(10, 10)
# # k = 0
# # for i in range(10):
# #     for j in range(10):
# #         ax[i][j].imshow(x_train[k].reshape(28, 28), aspect='auto')
# #         k += 1
# # plt.show()

# # Building the Sequential neural network model
# model = Sequential([
#     # Flatten input from 28x28 images to 784 (28*28) vector
#     Flatten(input_shape=(28, 28)),
  
#     # Dense layer 1 (256 neurons)
#     Dense(256, activation='sigmoid'),  
  
#     # Dense layer 2 (128 neurons)
#     Dense(128, activation='sigmoid'), 
  
#     # Output layer (10 classes)
#     Dense(10, activation='sigmoid'),  
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=10, 
#           batch_size=2000, 
#           validation_split=0.2)

# # Evaluating the model on test data
# results = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss, Test accuracy:', results)          

