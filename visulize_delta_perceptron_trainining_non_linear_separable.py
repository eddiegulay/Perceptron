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


def delta_rule_training_batch(X, labels, lr=0.01, epochs=50):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01  # Small random initialization
    targets = np.where(labels == 1, 1, -1)  # Convert labels to +1 and -1
    weight_history = []

    for epoch in range(epochs):
        y = np.dot(weights, X.T)  # Compute predictions for all samples
        error = targets - y       # Compute error for all samples
        grad = -np.dot(error, X) / n_samples  # Gradient over the entire batch
        weights -= lr * grad      # Update weights using the batch gradient
        weight_history.append(weights.copy())

    return weight_history

# Perceptron training
def perceptron_training(X, labels, lr=0.01, epochs=50):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01  # Small random initialization
    targets = np.where(labels == 1, 1, -1)  # Convert labels to +1 and -1
    weight_history = []

    for epoch in range(epochs):
        for i in range(n_samples):
            y = np.dot(weights, X[i])
            if targets[i] * y <= 0:  # Misclassification
                weights += lr * targets[i] * X[i]
        weight_history.append(weights.copy())

    return weight_history

# Plot animation
def animate_decision_boundary(X, labels, delta_history, perceptron_history, classA, classB, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot for data
    ax.scatter(classA[0], classA[1], c='red', label='Class A', alpha=0.7)
    ax.scatter(classB[0], classB[1], c='blue', label='Class B', alpha=0.7)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()

    # Initialize decision boundary lines
    delta_line, = ax.plot([], [], 'g-', label='Delta Rule')
    perceptron_line, = ax.plot([], [], 'orange', label='Perceptron')
    ax.legend(loc='upper left')

    # Add text annotations for accuracy
    delta_accuracy_text = ax.text(-1.9, -1.8, '', color='green', fontsize=10, ha='left')
    perceptron_accuracy_text = ax.text(-1.9, -1.6, '', color='orange', fontsize=10, ha='left')

    def calculate_accuracies(weights):
        if weights[1] != 0:  # Avoid division by zero
            predictions = np.sign(np.dot(weights, X.T))  # Make predictions
            classA_correct = np.sum((predictions == -1) & (labels == 0)) / np.sum(labels == 0) * 100
            classB_correct = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1) * 100
        else:
            classA_correct, classB_correct = 0, 0  # Invalid weights
        return classA_correct, classB_correct

    def update(frame):
        if frame < len(delta_history):
            delta_weights = delta_history[frame]
            if delta_weights[1] != 0:  # Avoid division by zero
                x1 = np.linspace(-2, 2, 100)
                x2 = (-delta_weights[0] * x1 - delta_weights[-1]) / delta_weights[1]
                delta_line.set_data(x1, x2)
            # Calculate accuracies
            delta_classA_acc, delta_classB_acc = calculate_accuracies(delta_weights)
            delta_accuracy_text.set_text(f"Delta: Class A ({delta_classA_acc:.1f}%), Class B ({delta_classB_acc:.1f}%)")

        if frame < len(perceptron_history):
            perceptron_weights = perceptron_history[frame]
            if perceptron_weights[1] != 0:  # Avoid division by zero
                x1 = np.linspace(-2, 2, 100)
                x2 = (-perceptron_weights[0] * x1 - perceptron_weights[-1]) / perceptron_weights[1]
                perceptron_line.set_data(x1, x2)
            # Calculate accuracies
            perceptron_classA_acc, perceptron_classB_acc = calculate_accuracies(perceptron_weights)
            perceptron_accuracy_text.set_text(f"Perceptron: Class A ({perceptron_classA_acc:.1f}%), Class B ({perceptron_classB_acc:.1f}%)")

        return delta_line, perceptron_line, delta_accuracy_text, perceptron_accuracy_text

    ani = FuncAnimation(fig, update, frames=max(len(delta_history), len(perceptron_history)),
                        interval=200, repeat=False)
    if title != "":
        plt.title(title)
    else:
        plt.title("Delta Rule vs Perceptron on Non-Linear Data")
    plt.show()


def subsample_data(classA, classB, scenario):
    """
    Subsample the data based on the scenario.
    """
    nA = classA.shape[1]
    nB = classB.shape[1]

    if scenario == "25% random from both classes":
        idxA = np.random.choice(nA, int(nA * 0.75), replace=False)
        idxB = np.random.choice(nB, int(nB * 0.75), replace=False)
        classA = classA[:, idxA]
        classB = classB[:, idxB]

    elif scenario == "50% random from classA":
        idxA = np.random.choice(nA, int(nA * 0.50), replace=False)
        classA = classA[:, idxA]

    elif scenario == "50% random from classB":
        idxB = np.random.choice(nB, int(nB * 0.50), replace=False)
        classB = classB[:, idxB]

    elif scenario == "20% from classA (x < 0), 80% from classA (x > 0)":
        idx_neg = np.where(classA[0, :] < 0)[0]
        idx_pos = np.where(classA[0, :] >= 0)[0]

        neg_sample = np.random.choice(idx_neg, int(len(idx_neg) * 0.80), replace=False)
        pos_sample = np.random.choice(idx_pos, int(len(idx_pos) * 0.20), replace=False)

        idxA = np.hstack((neg_sample, pos_sample))
        classA = classA[:, idxA]

    return classA, classB


def generate_new_dataset():
    """
    Generate the new non-linearly separable dataset.
    """
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


# Subsample and Evaluate
def evaluate_subsampling():
    scenarios = [
        "25% random from both classes",
        "50% random from classA",
        "50% random from classB",
        "20% from classA (x < 0), 80% from classA (x > 0)"
    ]

    X, X_bias, labels, classA, classB = generate_new_dataset()

    for scenario in scenarios:
        print(f"Scenario: {scenario}")
        classA_sub, classB_sub = subsample_data(classA, classB, scenario)

        # Combine subsampled data
        X_sub = np.hstack((classA_sub, classB_sub)).T
        X_sub_bias = np.hstack((X_sub, np.ones((X_sub.shape[0], 1))))
        labels_sub = np.hstack((np.zeros(classA_sub.shape[1]), np.ones(classB_sub.shape[1])))

        # Shuffle data
        shuffle_idx = np.random.permutation(labels_sub.size)
        X_sub_bias = X_sub_bias[shuffle_idx]
        labels_sub = labels_sub[shuffle_idx]

        # Train both models
        delta_history = delta_rule_training_batch(X_sub_bias, labels_sub, lr=0.01, epochs=50)
        perceptron_history = perceptron_training(X_sub_bias, labels_sub, lr=0.01, epochs=50)

        # Visualize decision boundary evolution
        animate_decision_boundary(X_sub_bias, labels_sub, delta_history, perceptron_history, classA_sub, classB_sub, title=scenario)


# Run the evaluation
evaluate_subsampling()


# Main script
X, X_bias, labels, classA, classB = generate_non_linearly_separable_data()

# Train using Delta Rule and Perceptron
delta_history = delta_rule_training_batch(X_bias, labels, lr=0.01, epochs=50)
perceptron_history = perceptron_training(X_bias, labels, lr=0.01, epochs=50)

# Animate decision boundaries
animate_decision_boundary(X_bias, labels, delta_history, perceptron_history, classA, classB)
