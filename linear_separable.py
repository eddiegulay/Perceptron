import numpy as np
import matplotlib.pyplot as plt

def generate_data(n=100, mA=(1.0, 0.5), sigmaA=0.5, mB=(-1.0, 0.0), sigmaB=0.5):
    np.random.seed(42)  # Ensures reproducibility
    classA = np.vstack((np.random.randn(n) * sigmaA + mA[0],
                        np.random.randn(n) * sigmaA + mA[1]))
    classB = np.vstack((np.random.randn(n) * sigmaB + mB[0],
                        np.random.randn(n) * sigmaB + mB[1]))
    
    X = np.hstack((classA, classB)).T
    X_bias = np.hstack((X, np.ones((2 * n, 1))))  # Add bias term
    labels = np.hstack((np.zeros(n), np.ones(n)))  # 0 for classA, 1 for classB
    
    shuffle_idx = np.random.permutation(2 * n)
    return X[shuffle_idx], X_bias[shuffle_idx], labels[shuffle_idx], classA, classB

def plot_data(classA, classB):
    plt.figure(figsize=(8, 5))
    plt.scatter(classA[0], classA[1], c='red', label='Class A')
    plt.scatter(classB[0], classB[1], c='blue', label='Class B')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Generated Data')
    plt.show()

def perceptron_learning(X, labels, lr=0.01, epochs=20):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01
    errors = []
    
    for epoch in range(epochs):
        error = 0
        for i in range(n_samples):
            x = X[i]
            y = np.dot(weights, x)
            pred = 1 if y >= 0 else 0
            if pred != labels[i]:
                error += 1
                weights += lr * (labels[i] - pred) * x
        errors.append(error / n_samples)
    return weights, errors

def delta_rule_sequential(X, labels, lr=0.001, epochs=20):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01
    mse_history = []
    
    # Convert labels to bipolar (-1, 1)
    targets = np.where(labels == 1, 1, -1)
    
    for epoch in range(epochs):
        mse = 0
        for i in range(n_samples):
            x = X[i]
            y = np.dot(weights, x)
            error = y - targets[i]
            mse += error**2
            weights -= lr * error * x
        mse_history.append(mse / n_samples)
    return weights, mse_history

def delta_rule_batch(X, labels, lr=0.001, epochs=20):
    n_samples, n_features = X.shape
    weights = np.random.randn(1, n_features) * 0.01  # Reshape weights to be 2D
    targets = np.where(labels == 1, 1, -1).reshape(1, -1)
    X_mat = X.T
    mse_history = []

    for epoch in range(epochs):
        y = np.dot(weights, X_mat)
        error = y - targets
        mse = np.mean(error**2)
        mse_history.append(mse)
        grad = np.dot(error, X_mat.T) / n_samples
        weights -= lr * grad
    return weights.flatten(), mse_history  # Flatten weights to return as 1D array

def plot_learning_curves(errors_perceptron, mse_seq, mse_batch, learning_rates):
    plt.figure(figsize=(12, 8))

    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 2, i + 1)
        plt.plot(errors_perceptron[i], label=f'Perceptron (lr={lr})', marker='o')
        plt.plot(mse_seq[i], label=f'Delta Seq (lr={lr})', marker='x')
        plt.plot(mse_batch[i], label=f'Delta Batch (lr={lr})', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Error/MSE')
        plt.legend()
        plt.title(f'Learning Curves (lr={lr})')

    plt.tight_layout()
    plt.show()

def plot_decision_boundaries(X, weights_list, labels_list, classA, classB):
    x1 = np.linspace(-3, 3, 100)
    plt.figure(figsize=(10, 6))

    for weights, label in zip(weights_list, labels_list):
        if weights is not None:
            x2 = (-weights[0] * x1 - weights[2]) / weights[1]
            plt.plot(x1, x2, label=label)

    plt.scatter(classA[0], classA[1], c='red', alpha=0.3, label='Class A')
    plt.scatter(classB[0], classB[1], c='blue', alpha=0.3, label='Class B')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Decision Boundaries')
    plt.show()

# Task 1: Generate linearly separable data
X, X_bias, labels, classA, classB = generate_data()
plot_data(classA, classB)

# Task 2: Perceptron and Delta Rule Sequential Learning with multiple learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
errors_perceptron_list = []
mse_seq_list = []
mse_batch_list = []

for lr in learning_rates:
    _, errors_perceptron = perceptron_learning(X_bias, labels, lr=lr, epochs=50)
    _, mse_seq = delta_rule_sequential(X_bias, labels, lr=lr, epochs=50)
    _, mse_batch = delta_rule_batch(X_bias, labels, lr=lr, epochs=50)
    
    errors_perceptron_list.append(errors_perceptron)
    mse_seq_list.append(mse_seq)
    mse_batch_list.append(mse_batch)

# Plot learning curves for all learning rates
plot_learning_curves(errors_perceptron_list, mse_seq_list, mse_batch_list, learning_rates)

# Plot decision boundaries for the first learning rate as an example
plot_decision_boundaries(X, [perceptron_learning(X_bias, labels, lr=learning_rates[0], epochs=50)[0],
                             delta_rule_sequential(X_bias, labels, lr=learning_rates[0], epochs=50)[0]],
                         [f'Perceptron (lr={learning_rates[0]})', f'Delta Rule (Seq, lr={learning_rates[0]})'],
                         classA, classB)
