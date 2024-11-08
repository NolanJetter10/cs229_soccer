import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
data = np.load("Processed_keypoints.npy")
data = data[:, :1, :, :]  # Only take the contact frame
data = np.reshape(data, (20, 50)).astype(np.float64)  # Ensure data is float64

labels = np.load("Data_labels.npy", allow_pickle=True)
labels = labels[:, 4:5]  # Focus on the type of strike (5th column)
labels = np.reshape(labels, (20,)).astype(np.float64)  # Ensure labels are float64

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient descent function for logistic regression
def gradient_descent(X, y, weights, alpha, num_iters):
    m = len(y)
    for _ in range(num_iters):
        predictions = sigmoid(X @ weights)
        gradient = (1 / m) * X.T @ (predictions - y)
        weights -= alpha * gradient
    return weights

# Leave-One-Out Cross-Validation with training and testing accuracy
def logistic_regression_loocv(data, labels, alpha, num_iters):
    m, n = data.shape
    cumulative_test_score = 0
    cumulative_train_score = 0

    for i in range(m):
        # Separate the test sample and training data for LOOCV
        X_train = np.delete(data, i, axis=0)
        y_train = np.delete(labels, i)
        X_test = data[i].reshape(1, -1)
        y_test = labels[i]

        # Initialize weights
        weights = np.zeros(n, dtype=np.float64)

        # Train the model using gradient descent
        weights = gradient_descent(X_train, y_train, weights, alpha, num_iters)

        # Make a prediction on the test sample
        test_prediction = sigmoid(X_test @ weights) >= 0.5
        cumulative_test_score += (test_prediction == y_test)

        # Calculate training accuracy on the remaining 19 samples
        train_predictions = sigmoid(X_train @ weights) >= 0.5
        cumulative_train_score += np.mean(train_predictions == y_train)

    # Calculate average accuracy across all LOOCV folds
    test_accuracy = cumulative_test_score / m
    train_accuracy = cumulative_train_score / m
    return test_accuracy, train_accuracy

# Parameters for gradient descent and iteration range
alpha = 0.01  # Learning rate
iteration_values = list(range(0, 100000, 5000))  # Range from 100 to 2000 by 100

# Lists to store accuracy results
test_accuracies = []
train_accuracies = []

# select the label: 

# Run LOOCV for each iteration count and record accuracies
for num_iters in iteration_values:
    test_accuracy, train_accuracy = logistic_regression_loocv(data, labels, alpha, num_iters)
    test_accuracies.append(test_accuracy)
    train_accuracies.append(train_accuracy)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(iteration_values, test_accuracies, label="Test Accuracy", color="blue", marker="o", linestyle="-")
plt.plot(iteration_values, train_accuracies, label="Train Accuracy", color="red", marker="o", linestyle="-")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy vs. Number of Iterations")
plt.legend()
plt.grid()
plt.show()
