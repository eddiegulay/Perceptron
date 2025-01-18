import numpy as np



class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.zeros(input_size + 1)
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x>=0 else 0

    def predict(self, x):
        # weighted sum plus bias
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        a = self.activation_fn(z)
        return a

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi , target in zip(X, y):
                prediction  = self.predict(xi)
                error = target - prediction

                #update weights and bias
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error


# Dataset
# AND Gate
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0,0,0,1])


# XOR Gate
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0,1,1,0])

# Training the model
print("Exploring Behavior of AND Gate on linearly separable data")
Perceptron_AND = Perceptron(input_size=2)
Perceptron_AND.train(X_and, y_and)

predictions = []
for i in X_and:
    predictions.append(Perceptron_AND.predict(i))
print("AND GATE\n",predictions,"\nActual\n",y_and)

# Test on XOR Gate
print("Exploring Behavior of XOR Gate on non-linearly separable data")
Perceptron_XOR = Perceptron(input_size=2)
Perceptron_XOR.train(X_xor, y_xor)

predictions = []
for i in X_xor:
    predictions.append(Perceptron_XOR.predict(i))
print("XOR GATE\n",predictions,"\nActual\n",y_xor)