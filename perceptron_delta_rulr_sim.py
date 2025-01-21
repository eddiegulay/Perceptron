import numpy as np
import matplotlib.pyplot as plt

# Generate linearly separable data
def generate_data(n=100, mA=[1.0, 0.5], sigmaA=0.5, mB=[-1.0, 0.0], sigmaB=0.5):
    classA = np.random.randn(2, n) * sigmaA + np.array(mA).reshape(-1, 1)
    classB = np.random.randn(2, n) * sigmaB + np.array(mB).reshape(-1, 1)
    data = np.hstack((classA, classB))
    labels = np.hstack((np.ones(n), -np.ones(n)))
    indices = np.random.permutation(2 * n)
    return data[:, indices], labels[indices]

# Initialize weights
def initialize_weights(input_dim):
    return np.random.randn(input_dim + 1) * 0.01

# Perceptron learning rule
def perceptron_learning(data, labels, learning_rate, epochs):
    weights = initialize_weights(data.shape[0])
    for epoch in range(epochs):
        for i in range(data.shape[1]):
            x = np.append(data[:, i], 1)  # Append bias term
            prediction = np.sign(np.dot(weights, x))
            error = labels[i] - prediction
            weights += learning_rate * error * x
    return weights

# Delta rule (sequential mode)
def delta_rule_sequential(data, labels, learning_rate, epochs):
    weights = initialize_weights(data.shape[0])
    for epoch in range(epochs):
        for i in range(data.shape[1]):
            x = np.append(data[:, i], 1)  # Append bias term
            output = np.dot(weights, x)
            error = labels[i] - output
            weights += learning_rate * error * x
    return weights

# Delta rule (batch mode)
def delta_rule_batch(data, labels, learning_rate, epochs):
    weights = initialize_weights(data.shape[0])
    x_augmented = np.vstack([data, np.ones(data.shape[1])])  # Add bias
    for epoch in range(epochs):
        outputs = np.dot(weights, x_augmented)
        errors = labels - outputs
        weight_update = learning_rate * np.dot(errors, x_augmented.T)
        weights += weight_update
    return weights

# Plot decision boundary
def plot_decision_boundary(data, labels, weights, title):
    plt.scatter(data[0, labels == 1], data[1, labels == 1], color='blue', label='Class A')
    plt.scatter(data[0, labels == -1], data[1, labels == -1], color='red', label='Class B')
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_values = np.linspace(x_min, x_max, 100)
    y_values = -(weights[0] * x_values + weights[2]) / weights[1]  # w1*x + w2*y + b = 0
    plt.plot(x_values, y_values, color='green', label='Decision Boundary')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Evaluate mean squared error
def mean_squared_error(data, labels, weights):
    x_augmented = np.vstack([data, np.ones(data.shape[1])])  # Add bias
    outputs = np.dot(weights, x_augmented)
    errors = labels - outputs
    return np.mean(errors**2)

LEARNING_RATES = [0.01, 0.1, 0.5, 1.0, 2.0]

# Main execution
def main():
    # Generate data
    data, labels = generate_data()
    
    for lr in LEARNING_RATES:

        # Perceptron learning
        weights_perceptron = perceptron_learning(data, labels, learning_rate=lr, epochs=20)
        plot_decision_boundary(data, labels, weights_perceptron, "Perceptron Learning: lr={}".format(lr))

        # Delta rule (sequential mode)
        weights_delta_seq = delta_rule_sequential(data, labels, learning_rate=lr, epochs=20)
        plot_decision_boundary(data, labels, weights_delta_seq, "Delta Rule - Sequential Mode: lr={}".format(lr))

        # Delta rule (batch mode)
        weights_delta_batch = delta_rule_batch(data, labels, learning_rate=lr, epochs=20)
        plot_decision_boundary(data, labels, weights_delta_batch, "Delta Rule - Batch Mode: lr={}".format(lr))

        # Compare learning curves
        x_augmented = np.vstack([data, np.ones(data.shape[1])])

        errors_seq = []
        weights = initialize_weights(data.shape[0])
        for epoch in range(20):
            for i in range(data.shape[1]):
                x = np.append(data[:, i], 1)
                output = np.dot(weights, x)
                error = labels[i] - output
                weights += 0.01 * error * x
            errors_seq.append(mean_squared_error(data, labels, weights))

        errors_batch = []
        weights = initialize_weights(data.shape[0])
        for epoch in range(20):
            outputs = np.dot(weights, x_augmented)
            errors = labels - outputs
            weight_update = 0.01 * np.dot(errors, x_augmented.T)
            weights += weight_update
            errors_batch.append(mean_squared_error(data, labels, weights))

        plt.plot(range(1, 21), errors_seq, label='Sequential Delta Rule', marker='o')
        plt.plot(range(1, 21), errors_batch, label='Batch Delta Rule', marker='x')
        plt.title("Learning Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
