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

# Batch Delta Rule Implementation Without Bias
def delta_rule_batch_no_bias(patterns, targets, learning_rate, epochs):
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

# Testing Perceptron Without Bias
def test_no_bias():
    patterns, targets = generate_data()
    learning_rate = 0.01
    epochs = 50

    weights, errors = delta_rule_batch_no_bias(patterns, targets, learning_rate, epochs)

    # Visualize learning curve
    plt.plot(range(epochs), errors, label='Batch Delta Rule Without Bias')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Delta Rule Learning Without Bias')
    plt.legend()
    plt.grid()
    plt.show()

    # Decision boundary
    plt.scatter(patterns[0, targets[0] == 1], patterns[1, targets[0] == 1], color='blue', label='Class A')
    plt.scatter(patterns[0, targets[0] == -1], patterns[1, targets[0] == -1], color='red', label='Class B')

    x_vals = np.linspace(-2, 2, 100)
    y_vals = -(weights[0, 0] / weights[0, 1]) * x_vals
    plt.plot(x_vals, y_vals, label='Decision Boundary', color='green')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundary Without Bias')
    plt.legend()
    plt.grid()
    plt.show()

    # Analysis of convergence and classification
    print("Final Weights:", weights)
    print("Final MSE:", errors[-1])

    return weights, errors

if __name__ == "__main__":
    test_no_bias()
