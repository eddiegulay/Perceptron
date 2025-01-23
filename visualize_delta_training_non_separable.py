import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Non-linear data generator
def generate_non_linearly_separable_data():
    ndata = 100
    mA = [1.0, 0.3]
    sigmaA = 0.2
    mB = [0.0, -0.1]
    sigmaB = 0.3

    classA_x = np.hstack((
        np.random.randn(round(0.5 * ndata)) * sigmaA - mA[0],
        np.random.randn(round(0.5 * ndata)) * sigmaA + mA[0]
    ))
    classA_y = np.random.randn(ndata) * sigmaA + mA[1]
    classB_x = np.random.randn(ndata) * sigmaB + mB[0]
    classB_y = np.random.randn(ndata) * sigmaB + mB[1]

    classA = np.vstack((classA_x, classA_y))
    classB = np.vstack((classB_x, classB_y))

    X = np.hstack((classA, classB)).T
    X_bias = np.hstack((X, np.ones((2 * ndata, 1))))
    labels = np.hstack((np.zeros(ndata), np.ones(ndata)))

    shuffle_idx = np.random.permutation(2 * ndata)
    return X[shuffle_idx], X_bias[shuffle_idx], labels[shuffle_idx], classA, classB

# Delta rule function with visualization data collection
def delta_rule_with_visualization(X, labels, lr=0.01, epochs=50):
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
    plt.title("Evolution of Decision Boundary on Non-Linear Data")
    plt.show()

# Main script
X, X_bias, labels, classA, classB = generate_non_linearly_separable_data()
weight_history = delta_rule_with_visualization(X_bias, labels, lr=0.01, epochs=50)
animate_decision_boundary(X_bias, labels, weight_history, classA, classB)
