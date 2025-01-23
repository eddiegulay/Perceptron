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

def perceptron_learning(X, labels, lr=0.1, epochs=20):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Initialize weights to zero
    errors = []

    for epoch in range(epochs):
        error = 0
        for i in range(n_samples):
            x = X[i]
            # Compute weighted sum
            y = np.dot(weights, x)
            # Perceptron prediction
            pred = 1 if y >= 0 else 0
            # Update weights if prediction is incorrect
            if pred != labels[i]:
                error += 1
                weights += lr * (labels[i] - pred) * x
        # Record the error rate for this epoch
        errors.append(error / n_samples)
    
    return weights, errors

def delta_rule_batch(X, labels, lr=0.001, epochs=20):
    n_samples, n_features = X.shape
    weights = np.random.randn(n_features) * 0.01  # Small random initialization
    targets = np.where(labels == 1, 1, -1)  # Convert labels to +1 and -1
    mse_history = []

    for epoch in range(epochs):
        # Compute predictions
        y = np.dot(weights, X.T)  # Weighted sum
        # Compute error
        error = targets - y
        # Compute MSE
        mse = np.mean(error**2)
        mse_history.append(mse)
        # Compute gradient
        grad = np.dot(error, X) / n_samples  # Note the negative sign
        # Update weights
        weights -= lr * grad
    
    return weights, mse_history

def subsample_data(X, labels, classA, classB, scenario):
    n = len(labels) // 2

    if scenario == "random_25_each":
        keep_indices = np.random.choice(n, int(0.75 * n), replace=False)
        classA_sub = classA[:, keep_indices]
        classB_sub = classB[:, keep_indices]

    elif scenario == "random_50_classA":
        keep_indices_A = np.random.choice(n, int(0.5 * n), replace=False)
        classA_sub = classA[:, keep_indices_A]
        classB_sub = classB

    elif scenario == "random_50_classB":
        keep_indices_B = np.random.choice(n, int(0.5 * n), replace=False)
        classA_sub = classA
        classB_sub = classB[:, keep_indices_B]

    elif scenario == "biased_classA":
        condition_A_neg = np.where(classA[0, :] < 0)[0]
        condition_A_pos = np.where(classA[0, :] >= 0)[0]

        neg_sample = np.random.choice(condition_A_neg, int(0.2 * len(condition_A_neg)), replace=False)
        pos_sample = np.random.choice(condition_A_pos, int(0.8 * len(condition_A_pos)), replace=False)
        keep_indices_A = np.hstack((neg_sample, pos_sample))
        classA_sub = classA[:, keep_indices_A]
        classB_sub = classB

    else:
        raise ValueError("Invalid scenario")

    X_sub = np.hstack((classA_sub, classB_sub)).T
    labels_sub = np.hstack((np.zeros(classA_sub.shape[1]), np.ones(classB_sub.shape[1])))
    shuffle_idx = np.random.permutation(len(labels_sub))
    return X_sub[shuffle_idx], labels_sub[shuffle_idx], classA_sub, classB_sub

def plot_data(classA, classB, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(classA[0], classA[1], c='red', label='Class A', alpha=0.7)
    plt.scatter(classB[0], classB[1], c='blue', label='Class B', alpha=0.7)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(title)
    plt.show()

def train_and_plot(X, labels, learning_rate, epochs, title):
    weights_perceptron, errors_perceptron = perceptron_learning(X, labels, lr=learning_rate, epochs=epochs)
    weights_delta, mse_delta = delta_rule_batch(X, labels, lr=learning_rate, epochs=epochs)

    x1 = np.linspace(-3, 3, 100)
    x2_perceptron = (-weights_perceptron[0] * x1 - weights_perceptron[-1]) / weights_perceptron[1]
    x2_delta = (-weights_delta[0] * x1 - weights_delta[-1]) / weights_delta[1]

    predictions_perceptron = np.dot(X, weights_perceptron) >= 0
    accuracy_A_perceptron = np.mean(predictions_perceptron[labels == 0] == 0)
    accuracy_B_perceptron = np.mean(predictions_perceptron[labels == 1] == 1)

    predictions_delta = np.sign(np.dot(X, weights_delta)) == 1
    accuracy_A_delta = np.mean(predictions_delta[labels == 0] == 0)
    accuracy_B_delta = np.mean(predictions_delta[labels == 1] == 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x1, x2_perceptron, 'r--', label=f'Perceptron (Class A: {accuracy_A_perceptron:.2f}, Class B: {accuracy_B_perceptron:.2f})')
    plt.plot(x1, x2_delta, 'b-', label=f'Delta Rule (Class A: {accuracy_A_delta:.2f}, Class B: {accuracy_B_delta:.2f})')
    plt.scatter(X[:, 0][labels == 0], X[:, 1][labels == 0], c='red', alpha=0.3, label='Class A')
    plt.scatter(X[:, 0][labels == 1], X[:, 1][labels == 1], c='blue', alpha=0.3, label='Class B')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(title)
    plt.show()

# Generate non-linearly separable data
X_nonlinear, X_bias_nonlinear, labels_nonlinear, classA_nl, classB_nl = generate_non_linearly_separable_data()
plot_data(classA_nl, classB_nl, "Non-Linearly Separable Data")

# Apply training and plot decision boundaries
train_and_plot(X_bias_nonlinear, labels_nonlinear, learning_rate=0.01, epochs=50, 
               title="Decision Boundaries for Non-Linearly Separable Data")

# Subsample data for different scenarios
scenarios = ["random_25_each", "random_50_classA", "random_50_classB", "biased_classA"]
for scenario in scenarios:
    X_sub, labels_sub, classA_sub, classB_sub = subsample_data(X_bias_nonlinear, labels_nonlinear, classA_nl, classB_nl, scenario)
    plot_data(classA_sub, classB_sub, f"Subsampled Data ({scenario})")
    train_and_plot(X_sub, labels_sub, learning_rate=0.01, epochs=50, 
                   title=f"Decision Boundaries for {scenario}")
