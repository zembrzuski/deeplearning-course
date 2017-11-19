import numpy as np


def sigmoid(x):
    """
    Computa a sigmoide de x. X pode ser um número real ou um numpy.array
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sig_x):
    """
    TODO
    """
    return sig_x * (1 - sig_x)


def loss(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


def cost(Y_hat, Y):
    return np.sum(loss(Y_hat, Y))/Y_hat.shape[1]


def forward_propagation(X, W1, b1, W2, b2):
    a0 = X

    z1 = np.dot(W1, a0) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    return {
        'a1': a1,
        'z1': z1,
        'a2': a2,
        'z2': z2
    }


def compute_gradient(X, Y, W1, b1, W2, b2, learning_rate):
    forward_prop = forward_propagation(X, W1, b1, W2, b2)

    m = X.shape[1]
    a1 = forward_prop['a1']
    a2 = forward_prop['a2']
    z1 = forward_prop['z1']
    dz2 = a2 - Y
    dw2 = (np.dot(dz2, a1.T))/m
    db2 = (np.sum(dz2, axis=1, keepdims=True))/m
    dz1 = np.dot(W2.T, dz2) * sigmoid_derivative(a1)
    dw1 = (np.dot(dz1, X.T))/m
    db1 = (np.sum(dz1, axis=1, keepdims=True)) / m

    kost = cost(a2, Y)

    # W1 = W1 - learning_rate * dw1
    # b1 = b1 - learning_rate * db1
    # W2 = W2 - learning_rate * dw2
    # b2 = b2 - learning_rate * db2

    return {
        'cost': kost,
        'dw2': dw2,
        'db2': db2,
        'dw1': dw1,
        'db1': db1
    }