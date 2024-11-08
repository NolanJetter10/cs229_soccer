import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the data
data = np.load("Processed_keypoints.npy")
data = data[:, :1, :, :]  # Only take the contact frame
data = np.reshape(data, (20, 50)).astype(np.float64)  # Ensure data is float64
labels_list = []
labels_names = ["direction", "height", "quality"]
labels = np.load("Data_labels.npy", allow_pickle=True)
labels_list.append(np.reshape(labels[:, 2:3], (20,)))
labels_list.append(np.reshape(labels[:, 3:4], (20,)))
quality_labels = np.reshape(labels[:, 5:6], (20,)) - 1
labels_list.append(quality_labels)
labels_list = np.array(labels_list)

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

# One-vs-Rest Logistic Regression with LOOCV
def logistic_regression_loocv_multiclass(data, labels, alpha, num_iters):
    m, n = data.shape
    cumulative_test_score = 0
    cumulative_train_score = 0

    for i in range(m):
        # Separate the test sample and training data for LOOCV
        X_train = np.delete(data, i, axis=0)
        y_train = np.delete(labels, i)
        X_test = data[i].reshape(1, -1)
        y_test = labels[i]

        # Initialize weights for each class in One-vs-Rest
        weights = np.zeros((3, n), dtype=np.float64)  # 3 classifiers for classes 0, 1, 2

        # Train a binary classifier for each class using One-vs-Rest
        for class_label in range(3):
            # Create binary labels for the current class
            binary_labels = (y_train == class_label).astype(np.float64)
            weights[class_label] = gradient_descent(X_train, binary_labels, weights[class_label], alpha, num_iters)

        # Predict on the test sample
        test_predictions = [sigmoid(X_test @ weights[class_label]) for class_label in range(3)]
        predicted_class = np.argmax(test_predictions)  # Choose the class with the highest probability
        cumulative_test_score += (predicted_class == y_test)

        # Calculate training accuracy on the remaining 19 samples
        train_predictions = [sigmoid(X_train @ weights[class_label]) >= 0.5 for class_label in range(3)]
        train_predicted_classes = np.argmax(train_predictions, axis=0)
        cumulative_train_score += np.mean(train_predicted_classes == y_train)

    # Calculate average accuracy across all LOOCV folds
    test_accuracy = cumulative_test_score / m
    train_accuracy = cumulative_train_score / m
    return test_accuracy, train_accuracy

# Parameters for gradient descent and iteration range
alpha = 0.01  # Learning rate
iteration_values = list(range(100, 10000, 1000))  # Range from 100 to 2000 by 100

# Lists to store accuracy results for each label type
test_list = []
train_list = []

for i in range(3):
    # Run LOOCV for each iteration count and record accuracies
    labels = labels_list[i]
    test_accuracies = []
    train_accuracies = []
    for num_iters in iteration_values:
        test_accuracy, train_accuracy = logistic_regression_loocv_multiclass(data, labels, alpha, num_iters)
        test_accuracies.append(test_accuracy)
        train_accuracies.append(train_accuracy)
    test_list.append(test_accuracies)  # Append the full list of test accuracies
    train_list.append(train_accuracies)  # Append the full list of train accuracies

for i in range(3):
    test_accuracies = test_list[i]
    train_accuracies = train_list[i]
    # Plotting results for each label type
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_values, test_accuracies, label="Test Accuracy", color="blue", marker="o", linestyle="-")
    plt.plot(iteration_values, train_accuracies, label="Train Accuracy", color="red", marker="o", linestyle="-")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Testing Accuracy vs. Number of Iterations for {labels_names[i].capitalize()}")
    plt.legend()
    plt.grid()
    plt.show()