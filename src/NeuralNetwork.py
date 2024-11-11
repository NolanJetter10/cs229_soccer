import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess the data
data = np.load("Processed_keypoints.npy")
data = data[:, :1, :, :]  # Only take the contact frame
data = np.reshape(data, (20, 50)).astype(np.float32)  # Reshape to (20, 50) and ensure float32 type

# Load and one-hot encode the labels for multi-class classification
labels = np.load("Data_labels.npy", allow_pickle=True)
labels_list = []
labels_names = ["direction", "height", "quality"]

# Prepare labels for each classification target
labels_list.append(to_categorical(np.reshape(labels[:, 2:3], (20,)), num_classes=3))
labels_list.append(to_categorical(np.reshape(labels[:, 3:4], (20,)), num_classes=3))
quality_labels = np.reshape(labels[:, 5:6], (20,)) - 1
labels_list.append(to_categorical(quality_labels, num_classes=3))

# Define neural network model structure
def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 output nodes for multi-class classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Parameters
num_epochs = 50
test_list = []
train_list = []

# Perform LOOCV for each label type
for i in range(3):
    labels = labels_list[i]
    m = data.shape[0]  # Number of samples
    epoch_test_accuracies = np.zeros((m, num_epochs))  # Store test accuracy per epoch per fold
    epoch_train_accuracies = np.zeros((m, num_epochs))  # Store train accuracy per epoch per fold

    # LOOCV: Train on 19 samples, test on the 1 left-out sample
    for j in range(m):
        # Split data for LOOCV
        X_train = np.delete(data, j, axis=0)
        y_train = np.delete(labels, j, axis=0)
        X_test = data[j].reshape(1, -1)
        y_test = labels[j].reshape(1, -1)

        # Create and train the model, using validation data for the left-out test sample
        model = create_model((50,))
        history = model.fit(
            X_train, y_train,
            epochs=num_epochs,
            verbose=0,
            validation_data=(X_test, y_test)
        )

        # Record training and test accuracy at each epoch
        epoch_train_accuracies[j] = history.history['accuracy']
        epoch_test_accuracies[j] = history.history['val_accuracy']

    # Average accuracies across all folds for each epoch
    avg_train_accuracy = np.mean(epoch_train_accuracies, axis=0)
    avg_test_accuracy = np.mean(epoch_test_accuracies, axis=0)

    # Store results for plotting
    train_list.append(avg_train_accuracy)
    test_list.append(avg_test_accuracy)

# Plot the results for each classification target
epoch_range = range(1, num_epochs + 1)
for i in range(3):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, test_list[i], label="Test Accuracy", color="blue", linestyle="-")
    plt.plot(epoch_range, train_list[i], label="Train Accuracy", color="red", linestyle="-")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Testing Accuracy per Epoch for {labels_names[i].capitalize()}")
    plt.legend()
    plt.grid()
    plt.show()