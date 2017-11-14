import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


def cost_function(a, y):
    i = loss(a, y)
    summmed = np.sum(i)
    return summmed / len(a)


def my_function(X, Y, W, b):
    J = 0; dw1 = 0; dw2 = 0; db = 0
    m = X.shape[1]

    for i in range(m):
        xi = X[:,i]
        yi = Y[i]
        zi = xi.dot(W) + b
        ai = sigmoid(zi)
        J += loss(ai, yi)
        dzi = ai - yi
        dw1 += xi[0] * dzi
        dw2 += xi[1] * dzi
        db += dzi


if __name__ == '__main__':
    point1 = np.array([1, 2])
    point2 = np.array([3, 5])
    point3 = np.array([4, 7])

    X = np.array([point1, point2, point3]).T
    Y = np.array([1, 1, 0])

    W = np.array([.5, .3])
    b = .2

    my_function(X, Y, W, b)
