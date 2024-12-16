import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

X = np.array([[0.66666667], [0.33333333], [1.0]])
y = np.array([[0.92], [0.86], [0.89]])

weights = np.random.rand(1, 1)
bias = np.random.rand(1)

lr = 0.1

for _ in range(10000):
    z = np.dot(X, weights) + bias
    output = sigmoid(z)

    error = y - output
    d_output = error * sigmoid_derivative(output)
    weights += np.dot(X.T, d_output) * lr
    bias += np.sum(d_output) * lr

print(f"Actual Output:\n{y}")
print(f"Predicted Output:\n{output}")
