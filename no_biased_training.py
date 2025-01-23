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

def delta_rule_batch_no_bias(X, labels, lr=0.001, epochs=20):
    n_samples, n_features = X.shape
    weights = np.random.randn(1, n_features - 1) * 0.01  # Ignore the bias term
    targets = np.where(labels == 1, 1, -1).reshape(1, -1)
    X_mat = X[:, :-1].T  # Exclude the bias column
    mse_history = []

    for epoch in range(epochs):
        y = np.dot(weights, X_mat)
        error = y - targets
        mse = np.mean(error**2)
        mse_history.append(mse)
        grad = np.dot(error, X_mat.T) / n_samples
        weights -= lr * grad
    return weights.flatten(), mse_history

def plot_decision_boundary_no_bias(X, weights, classA, classB, title):
    x1 = np.linspace(-3, 3, 100)
    x2 = (-weights[0] * x1) / weights[1]

    plt.figure(figsize=(10, 6))
    plt.plot(x1, x2, label='Decision Boundary (No Bias)')
    plt.scatter(classA[0], classA[1], c='red', alpha=0.3, label='Class A')
    plt.scatter(classB[0], classB[1], c='blue', alpha=0.3, label='Class B')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(title)
    plt.show()


X, X_bias, labels, classA, classB = generate_data()
plot_data(classA, classB)

# Train the network using delta rule without bias
weights_no_bias, mse_no_bias = delta_rule_batch_no_bias(X_bias, labels, lr=0.01, epochs=50)

# Plot decision boundary for the delta rule without bias
plot_decision_boundary_no_bias(X, weights_no_bias, classA, classB, 
                               title='Delta Rule Without Bias - Decision Boundary')


mA_new = (0.5, 0.5)
mB_new = (-0.5, -0.5)
X_new, X_bias_new, labels_new, classA_new, classB_new = generate_data(mA=mA_new, mB=mB_new)
plot_data(classA_new, classB_new)


weights_no_bias_new, mse_no_bias_new = delta_rule_batch_no_bias(X_bias_new, labels_new, lr=0.01, epochs=50)
plot_decision_boundary_no_bias(X_new, weights_no_bias_new, classA_new, classB_new, 
                               title='Delta Rule Without Bias - Adjusted Data Parameters')
