# Classification of Non-Linearly Separable Data

import numpy as np
import matplotlib.pyplot as plt

# Generate data that is not linearly separable
def generate_non_linear_data(ndata=100, mA=[1.0, 0.3], sigmaA=0.2, mB=[0.0, -0.1], sigmaB=0.3):
    classA = np.zeros((2, ndata))
    classB = np.zeros((2, ndata))

    # Generate class A data
    half_data = round(0.5 * ndata)
    classA[0, :half_data] = np.random.randn(half_data) * sigmaA - mA[0]
    classA[0, half_data:] = np.random.randn(half_data) * sigmaA + mA[0]
    classA[1, :] = np.random.randn(ndata) * sigmaA + mA[1]

    # Generate class B data
    classB[0, :] = np.random.randn(ndata) * sigmaB + mB[0]
    classB[1, :] = np.random.randn(ndata) * sigmaB + mB[1]

    return classA, classB

# Plot data
def plot_data(classA, classB):
    plt.figure(figsize=(8, 6))
    plt.scatter(classA[0, :], classA[1, :], c='blue', label='Class A')
    plt.scatter(classB[0, :], classB[1, :], c='red', label='Class B')
    plt.legend()
    plt.title('Non-Linearly Separable Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.show()

# Delta Rule Training in Batch Mode
def delta_rule_batch(X, T, epochs=20, learning_rate=0.01):
    num_features, num_samples = X.shape
    num_classes = T.shape[0]

    W = np.random.randn(num_classes, num_features) * 0.01
    errors = []

    for epoch in range(epochs):
        Y = np.dot(W, X)
        delta_W = -learning_rate * np.dot((np.dot(W, X) - T), X.T)
        W += delta_W
        error = np.mean((T - Y) ** 2)
        errors.append(error)

    return W, errors

# Subsample Data
def subsample_data(X, T, scenario):
    ndata = X.shape[1] // 2
    if scenario == 'random_25_percent':
        indices = np.random.choice(range(X.shape[1]), size=int(0.75 * X.shape[1]), replace=False)
    elif scenario == 'random_50_classA':
        classA_indices = np.random.choice(range(ndata), size=int(0.5 * ndata), replace=False)
        indices = np.concatenate((classA_indices, range(ndata, 2 * ndata)))
    elif scenario == 'random_50_classB':
        classB_indices = np.random.choice(range(ndata, 2 * ndata), size=int(0.5 * ndata), replace=False)
        indices = np.concatenate((range(ndata), classB_indices))
    elif scenario == 'subset_classA':
        classA_indices = np.where(X[0, :ndata] < 0)[0]
        classA_sample = np.random.choice(classA_indices, size=int(0.2 * len(classA_indices)), replace=False)
        other_classA = np.random.choice(np.where(X[0, :ndata] >= 0)[0], size=int(0.8 * len(classA_indices)), replace=False)
        indices = np.concatenate((classA_sample, other_classA, range(ndata, 2 * ndata)))
    else:
        raise ValueError('Invalid scenario')

    X_subsampled = X[:, indices]
    T_subsampled = T[:, indices]
    return X_subsampled, T_subsampled

# Main Execution
ndata = 100
mA = [1.0, 0.3]
sigmaA = 0.2
mB = [0.0, -0.1]
sigmaB = 0.3

classA, classB = generate_non_linear_data(ndata, mA, sigmaA, mB, sigmaB)
plot_data(classA, classB)

# Combine data and labels
X = np.hstack((classA, classB))
T = np.hstack((np.ones((1, ndata)), -np.ones((1, ndata))))

# Train with delta rule batch mode
W, errors = delta_rule_batch(X, T)
plt.plot(errors)
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# Subsample and evaluate scenarios
scenarios = ['random_25_percent', 'random_50_classA', 'random_50_classB', 'subset_classA']
for scenario in scenarios:
    X_sub, T_sub = subsample_data(X, T, scenario)
    W_sub, errors_sub = delta_rule_batch(X_sub, T_sub)
    print(f"Scenario: {scenario}, Final Error: {errors_sub[-1]:.4f}")
