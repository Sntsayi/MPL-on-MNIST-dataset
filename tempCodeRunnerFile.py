import numpy as np
# from tensorflow.keras.datasets import mnist
# import os

# # Define paths
# # NPZ is a file format by numpy that provides storage of array data using gzip compression
# model_path = "model_parameters.npz"

# # Load the dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Preprocessing: Flatten images, normalize, and one-hot encode labels
# x_train = x_train.reshape(len(x_train), 28 * 28) / 255.0  
# x_test = x_test.reshape(len(x_test), 28 * 28) / 255.0
# num_classes = 10
# y_train_onehot = np.eye(num_classes)[y_train]
# y_test_onehot = np.eye(num_classes)[y_test]

# # Define MLP Architecture
# input_size = 28 * 28
# hidden_size = 128
# output_size = 10

# # Define training parameters
# epochs = 10  # Ensure that `epochs` is always defined
# learning_rate = 0.01  # Define learning rate

# # Activation functions
# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return (x > 0).astype(float)

# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# # Check if model parameters already exist....amazing😍😎
# if os.path.exists(model_path):
#     print("Model parameters file found. Loading model...")
#     # Load the saved weights and biases
#     data = np.load(model_path)
#     W1 = data['W1']
#     b1 = data['b1']
#     W2 = data['W2']
#     b2 = data['b2']
#     print("Model parameters loaded successfully!")
# else:
#     print("Model parameters file not found. Training the model...")
#     # Initialize weights and biases (only if the model is not pre-trained)
#     W1 = np.random.randn(input_size, hidden_size) * 0.01
#     b1 = np.zeros((1, hidden_size))
#     W2 = np.random.randn(hidden_size, output_size) * 0.01
#     b2 = np.zeros((1, output_size))

#     # Training loop
#     for epoch in range(epochs):
#         # Training step
#         for i in range(x_train.shape[0]):
#             x_sample = x_train[i:i + 1]
#             y_sample = y_train_onehot[i:i + 1]

#             # Forward pass
#             z1 = np.dot(x_sample, W1) + b1
#             a1 = relu(z1)
#             z2 = np.dot(a1, W2) + b2
#             a2 = softmax(z2)

#             # Backward pass ==> "i got this backward pass from chatGpt"
#             dz2 = a2 - y_sample
#             dW2 = np.dot(a1.T, dz2)
#             db2 = np.sum(dz2, axis=0, keepdims=True)

#             dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)
#             dW1 = np.dot(x_sample.T, dz1)
#             db1 = np.sum(dz1, axis=0, keepdims=True)

#             # Update weights and biases
#             W2 -= learning_rate * dW2
#             b2 -= learning_rate * db2
#             W1 -= learning_rate * dW1
#             b1 -= learning_rate * db1

#         # Calculate training accuracy after each epoch
#         z1_train = np.dot(x_train, W1) + b1
#         a1_train = relu(z1_train)
#         z2_train = np.dot(a1_train, W2) + b2
#         a2_train = softmax(z2_train)
#         train_predictions = np.argmax(a2_train, axis=1)
#         train_accuracy = np.mean(train_predictions == y_train)

#         print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy * 100:.2f}%")

#     # Save the weights and biases after training....so amazing thing from numpy😍😍
#     np.savez(model_path, W1=W1, b1=b1, W2=W2, b2=b2)
#     print("Model parameters saved successfully!")

# # Test the model with the loaded or trained parameters
# z1_test = np.dot(x_test, W1) + b1
# a1_test = relu(z1_test)
# z2_test = np.dot(a1_test, W2) + b2
# a2_test = softmax(z2_test)
# test_predictions = np.argmax(a2_test, axis=1)
# test_accuracy = np.mean(test_predictions == y_test)

# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
