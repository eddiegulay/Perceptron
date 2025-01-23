import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Data generation function (reuse your own function or this minimal example)
def generate_data(n=100, mA=(1.0, 0.5), sigmaA=0.5, mB=(-1.0, 0.0), sigmaB=0.5):
    np.random.seed(42)  # Ensures reproducibility
    classA = np.vstack((np.random.randn(n) * sigmaA + mA[0],
                        np.random.randn(n) * sigmaA + mA[1]))
    classB = np.vstack((np.random.randn(n) * sigmaB + mB[0],
                        np.random.randn(n) * sigmaB + mB[1]))
    X = np.hstack((classA, classB)).T
    X_bias = np.hstack((X, np.ones((2 * n, 1))))  # Add bias term
    labels = np.hstack((np.zeros(n), np.ones(n)))  # 0 for classA, 1 for classB
    return X_bias, labels, classA, classB

# Delta rule function with visualization data collection
def delta_rule_with_visualization(X, labels, lr=0.001, epochs=20):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01  # Small random initialization
    targets = np.where(labels == 1, 1, -1)  # Convert labels to +1 and -1
    weight_history = []  # To store weights at each epoch
    
    for epoch in range(epochs):
        y = np.dot(weights, X.T)  # Weighted sum
        error = targets - y  # Compute error
        grad = -np.dot(error, X) / n_samples  # Compute gradient
        weights -= lr * grad  # Update weights
        weight_history.append(weights.copy())  # Save weights for visualization

    return weight_history

# Plot animation
def animate_decision_boundary(X, labels, weight_history, classA, classB):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot for data
    ax.scatter(classA[0], classA[1], c='red', label='Class A', alpha=0.7)
    ax.scatter(classB[0], classB[1], c='blue', label='Class B', alpha=0.7)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()

    # Initialize decision boundary line
    line, = ax.plot([], [], 'g-', label='Decision Boundary')
    ax.legend(loc='upper left')

    def update(frame):
        weights = weight_history[frame]
        if weights[1] != 0:  # Avoid division by zero
            x1 = np.linspace(-2, 2, 100)
            x2 = (-weights[0] * x1 - weights[-1]) / weights[1]  # Decision boundary formula
            line.set_data(x1, x2)
        return line,

    ani = FuncAnimation(fig, update, frames=len(weight_history), interval=200, repeat=True)
    plt.title("Evolution of Decision Boundary During Training")
    plt.show()

# Main script
X, labels, classA, classB = generate_data()
weight_history = delta_rule_with_visualization(X, labels, lr=0.01, epochs=50)
animate_decision_boundary(X, labels, weight_history, classA, classB)
