import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
n = 100
mA, sigmaA = [1.0, 0.5], 0.5
mB, sigmaB = [-1.0, 0.0], 0.5

classA = np.vstack((np.random.randn(n)*sigmaA + mA[0], 
                    np.random.randn(n)*sigmaA + mA[1]))
classB = np.vstack((np.random.randn(n)*sigmaB + mB[0], 
                    np.random.randn(n)*sigmaB + mB[1]))

# Combine, shuffle, and add bias term (ones)
X = np.hstack((classA, classB)).T  # Shape: (200, 2)
X_bias = np.hstack((X, np.ones((2*n, 1))))  # Add bias term (third column=1)
labels = np.hstack((np.zeros(n), np.ones(n)))  # 0 for classA, 1 for classB

shuffle_idx = np.random.permutation(2*n)
X, X_bias, labels = X[shuffle_idx], X_bias[shuffle_idx], labels[shuffle_idx]

# Plot data
plt.figure(figsize=(8, 5))
plt.scatter(classA[0], classA[1], c='red', label='Class A')
plt.scatter(classB[0], classB[1], c='blue', label='Class B')
plt.xlabel('x1'), plt.ylabel('x2'), plt.legend()
plt.title('Generated Data (Linearly Separable)')
plt.show()