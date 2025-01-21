import numpy as np
import matplotlib.pyplot as plt

# Data Generation (3.1.1)
def generate_data():
    n = 100
    mA, sigmaA = [1.0, 0.5], 0.5
    mB, sigmaB = [-1.0, 0.0], 0.5

    classA = np.random.randn(2, n) * sigmaA + np.array(mA).reshape(-1, 1)
    classB = np.random.randn(2, n) * sigmaB + np.array(mB).reshape(-1, 1)

    patterns = np.hstack((classA, classB))
    targets = np.hstack((np.ones(n), -np.ones(n)))

    shuffle_indices = np.random.permutation(2 * n)
    patterns, targets = patterns[:, shuffle_indices], targets[shuffle_indices]

    return patterns, targets

# Sequential Delta Rule Implementation
def delta_rule_sequential(patterns, targets, learning_rate, epochs):
    patterns = np.vstack((patterns, np.ones(patterns.shape[1])))
    weights = np.random.randn(1, patterns.shape[0]) * 0.01
    errors = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(patterns.shape[1]):
            x = patterns[:, i:i+1]
            t = targets[i]
            error = t - weights @ x
            weights += learning_rate * error * x.T
            total_error += error**2
        errors.append(total_error[0, 0] / patterns.shape[1])

    return weights, errors

# Batch Delta Rule Implementation
def delta_rule_batch(patterns, targets, learning_rate, epochs):
    patterns = np.vstack((patterns, np.ones(patterns.shape[1])))
    weights = np.random.randn(1, patterns.shape[0]) * 0.01
    targets = targets.reshape(1, -1)
    errors = []

    for epoch in range(epochs):
        outputs = weights @ patterns
        error = targets - outputs
        weights += learning_rate * (error @ patterns.T) / patterns.shape[1]
        mse = np.mean(error**2)
        errors.append(mse)

    return weights, errors

# Comparison of Sequential and Batch Learning
def compare_delta_rules():
    patterns, targets = generate_data()
    learning_rate = 0.1
    epochs = 50

    _, errors_sequential = delta_rule_sequential(patterns, targets, learning_rate, epochs)
    _, errors_batch = delta_rule_batch(patterns, targets, learning_rate, epochs)

    plt.plot(range(epochs), errors_sequential, label='Sequential Delta Rule')
    plt.plot(range(epochs), errors_batch, label='Batch Delta Rule')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Comparison of Sequential and Batch Delta Rule Learning')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    compare_delta_rules()

    # Squential Delta Rule learning converge faster with smaller learning rate <> Batch Delta Rule learning