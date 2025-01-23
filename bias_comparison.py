import numpy as np
import matplotlib.pyplot as plt


def generate_data(n=100, mA=(1.0, 0.5), sigmaA=0.5, mB=(-1.0, 0.0), sigmaB=0.5, include_bias=True):
    np.random.seed(42)
    classA = np.vstack((np.random.randn(n) * sigmaA + mA[0],
                        np.random.randn(n) * sigmaA + mA[1]))
    classB = np.vstack((np.random.randn(n) * sigmaB + mB[0],
                        np.random.randn(n) * sigmaB + mB[1]))
    
    X = np.hstack((classA, classB)).T
    if include_bias:
        X = np.hstack((X, np.ones((2 * n, 1))))  # Add bias term
    labels = np.hstack((np.zeros(n), np.ones(n)))
    
    shuffle_idx = np.random.permutation(2 * n)
    return X[shuffle_idx], labels[shuffle_idx], classA, classB

def delta_rule_batch(X, labels, lr=0.001, epochs=20):
    n_samples, n_features = X.shape
    weights = np.random.randn(1, n_features) * 0.01
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
    return weights.flatten(), mse_history

def test_bias_effect(mA, mB, include_bias=True):
    # Generate data with/without bias
    X, labels, classA, classB = generate_data(mA=mA, mB=mB, include_bias=include_bias)
    
    # Train with delta rule (batch)
    weights, mse_history = delta_rule_batch(X, labels, lr=0.01, epochs=50)
    
    # Plot decision boundary
    plt.figure(figsize=(8, 5))
    plt.scatter(classA[0], classA[1], c='red', alpha=0.3, label='Class A')
    plt.scatter(classB[0], classB[1], c='blue', alpha=0.3, label='Class B')
    
    if include_bias:
        x1 = np.linspace(-3, 3, 100)
        x2 = (-weights[0] * x1 - weights[2]) / weights[1]  # Include bias weight
    else:
        x1 = np.linspace(-3, 3, 100)
        x2 = (-weights[0] * x1) / weights[1]  # No bias term
    
    plt.plot(x1, x2, 'k-', label='Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(f'Bias={include_bias}, mA={mA}, mB={mB}')
    plt.show()

    # Plot MSE history
    plt.figure(figsize=(8, 5))
    plt.plot(mse_history, label='MSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'MSE History (Bias={include_bias}, mA={mA}, mB={mB})')
    plt.legend()
    plt.show()
    
    return mse_history

# Case 1: Data Separable Through Origin (Bias Not Needed)
# With Bias
mse_bias_origin = test_bias_effect(mA=(1.0, 1.0), mB=(-1.0, -1.0), include_bias=True)
# Without Bias
mse_nobias_origin = test_bias_effect(mA=(1.0, 1.0), mB=(-1.0, -1.0), include_bias=False)

# Case 2: Data Separable But Requires Offset (Bias Needed)
# With Bias
mse_bias_offset = test_bias_effect(mA=(5.0, 0.5), mB=(2.0, 0.0), include_bias=True)
# Without Bias
mse_nobias_offset = test_bias_effect(mA=(5.0, 0.5), mB=(2.0, 0.0), include_bias=False)

# Case 3: Non-Linear Separation (Both Fail)
# With Bias
mse_bias_nonlinear = test_bias_effect(mA=(0.5, 0.5), mB=(0.5, -0.5), include_bias=True)
# Without Bias
mse_nobias_nonlinear = test_bias_effect(mA=(0.5, 0.5), mB=(0.5, -0.5), include_bias=False)
