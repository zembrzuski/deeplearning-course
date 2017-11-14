import numpy as np


def loss(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


def cost_function():
    print("implemente isso, rapaz")


x1 = np.array([2, 3])
x2 = np.array([5, -2])

X = np.array([x1, x2]).T
W = np.array([1, 4])
b = 5

print(X)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(x1, x2, w1, w2, b):
    z = w1 * x1 + w2 * x2 + b
    a = sigmoid(z)
    return a


def forward_propagation_vectorized(X, W, b):
    z = X.dot(W) + b
    a = sigmoid(z)
    return a


a1 = forward_propagation_vectorized(x1, W, b)
a2 = forward_propagation_vectorized(x2, W, b)
print('a1: ' + str(a1))
print('a2: ' + str(a2))

y1 = 0
y2 = 0
Y = np.array([y1, y2])

def loss(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))

print('loss 1 :' + str(loss(a1, y1)))
print('loss 2 :' + str(loss(a2, y2)))

A = np.array([a1, a2])

print(loss(A, Y))


# aqui, recebe os arrays
def cost_function(a, y):
    i = loss(a, y)
    summmed = np.sum(i)
    return summmed / len(a)


print(cost_function(A, Y))
