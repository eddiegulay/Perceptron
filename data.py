import numpy as np
import matplotlib.pyplot as plt

# Parameters for class A
n = 100  # Number of points per class
mA = [1.0, 0.5]  # Mean of class A
sigmaA = 0.5  # Standard deviation of class A

# Parameters for class B
mB = [-1.0, 0.0]  # Mean of class B
sigmaB = 0.5  # Standard deviation of class B

# Generate data for class A
classA = np.zeros((2, n))
classA[0, :] = np.random.randn(n) * sigmaA + mA[0]
classA[1, :] = np.random.randn(n) * sigmaA + mA[1]

# Generate data for class B
classB = np.zeros((2, n))
classB[0, :] = np.random.randn(n) * sigmaB + mB[0]
classB[1, :] = np.random.randn(n) * sigmaB + mB[1]

# Combine and shuffle the data
data = np.hstack((classA, classB))  # Combine both classes
labels = np.hstack((np.ones(n), -np.ones(n)))  # Labels: +1 for class A, -1 for class B

# Shuffle the data
indices = np.random.permutation(2 * n)
data = data[:, indices]
labels = labels[indices]

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(classA[0, :], classA[1, :], color='blue', label='Class A')
plt.scatter(classB[0, :], classB[1, :], color='red', label='Class B')
plt.title("Linearly Separable Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
