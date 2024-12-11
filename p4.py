import numpy as np

X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float) / np.amax([[2, 9], [1, 5], [3, 6]], axis=0)
y = np.array([[92], [86], [89]], dtype=float) / 100

sigmoid = lambda x: 1 / (1 + np.exp(-x))
d_sigmoid = lambda x: x * (1 - x)

wh, bh = np.random.uniform(size=(2, 3)), np.random.uniform(size=(1, 3))
wo, bo = np.random.uniform(size=(3, 1)), np.random.uniform(size=(1, 1))

for _ in range(1000):
    h = sigmoid(X @ wh + bh)
    o = sigmoid(h @ wo + bo)
    d_o = (y - o) * d_sigmoid(o)
    d_h = d_o @ wo.T * d_sigmoid(h)
    wo += h.T @ d_o * 0.6
    wh += X.T @ d_h * 0.6

print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", o)