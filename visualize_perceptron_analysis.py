import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate dataset
def generate_data(n, mA, mB, sigma):
    classA = np.random.multivariate_normal(mA, sigma*np.identity(2), n)
    classB = np.random.multivariate_normal(mB, sigma*np.identity(2), n)
    
    X = np.vstack((classA, classB))
    y = np.hstack((np.ones(n), -np.ones(n)))
    return X, y

# Activation functions
def perceptron_activation(output):
    return np.where(output > 0, 1, -1)

def delta_activation(output):
    return output

# Perceptron learning rule (online)
def perceptron_training(X, y, learning_rate, epochs, add_bias=True):
    if add_bias:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    
    W = np.random.randn(X.shape[1])
    errors = []
    weights = [W.copy()]

    for _ in range(epochs):
        error = 0
        for i in range(len(X)):
            output = np.dot(W, X[i])
            pred = perceptron_activation(output)
            
            if pred != y[i]:
                W += learning_rate * y[i] * X[i]
                error += 1
        errors.append(error / len(X))
        weights.append(W.copy())
    
    return W, errors, weights

# Delta rule (batch)
def delta_training_batch(X, y, learning_rate, epochs, add_bias=True):
    if add_bias:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    
    W = np.random.randn(X.shape[1])
    errors = []
    weights = [W.copy()]

    for _ in range(epochs):
        output = np.dot(X, W)
        error = y - delta_activation(output)
        mse = np.mean(error ** 2)
        W += learning_rate * np.dot(X.T, error) / len(X)
        errors.append(mse)
        weights.append(W.copy())

    return W, errors, weights

# Delta rule (sequential)
def delta_training_sequential(X, y, learning_rate, epochs, add_bias=True):
    if add_bias:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    
    W = np.random.randn(X.shape[1])
    errors = []
    weights = [W.copy()]

    for _ in range(epochs):
        total_error = 0
        for i in range(len(X)):
            output = np.dot(W, X[i])
            error = y[i] - delta_activation(output)
            W += learning_rate * error * X[i]
            total_error += error ** 2
        mse = total_error / len(X)
        errors.append(mse)
        weights.append(W.copy())

    return W, errors, weights

# Helper function to plot decision boundary
def plot_decision_boundary(ax, W, add_bias=True):
    x_vals = np.linspace(-2, 2, 100)
    if add_bias:
        y_vals = -(W[0] * x_vals + W[2]) / W[1]
    else:
        y_vals = -(W[0] * x_vals) / W[1]
    ax.plot(x_vals, y_vals, 'k--')

# Helper function to animate the learning process
def animate_learning(ax, weights, X, y, add_bias):
    def update(frame):
        ax.clear()
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Class A')
        ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='red', label='Class B')
        ax.legend()
        plot_decision_boundary(ax, weights[frame], add_bias)
        ax.set_title(f"Epoch {frame + 1}")

    ani = FuncAnimation(fig=ax.figure, func=update, frames=len(weights), repeat=False)
    return ani

# Main function to create 2x2 plot
n, mA, mB, sigma = 100, [-1.0, 0.3], [1.0, -0.1], 0.2
X, y = generate_data(n, mA, mB, sigma)
epochs = 20
learning_rate = 0.1

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Training with bias
_, perceptron_errors_bias, perceptron_weights_bias = perceptron_training(X, y, learning_rate, epochs, add_bias=True)
_, delta_batch_errors_bias, delta_batch_weights_bias = delta_training_batch(X, y, learning_rate, epochs, add_bias=True)

animate_learning(axes[0, 0], perceptron_weights_bias, X, y, add_bias=True)
axes[0, 0].set_title("Perceptron (with bias)")
axes[0, 1].plot(range(1, epochs + 1), perceptron_errors_bias, label="MSE")
axes[0, 1].set_title("Error Curve (Perceptron with bias)")

animate_learning(axes[1, 0], delta_batch_weights_bias, X, y, add_bias=True)
axes[1, 0].set_title("Delta Batch Rule (with bias)")

plt.show()