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


def create_a_nn_layer(number_of_neurons, number_of_features):
    W = np.random.rand(number_of_neurons, number_of_features)
    B = np.random.rand(number_of_neurons, 1)

    return W, B


def forward_propagation_for_three_layered_neural_net(X, W1, b1, W2, b2, W3, b3, g1, g2, g3):
    a0 = X

    z1 = np.dot(W1, a0) + b1
    a1 = g1(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = g2(z2)

    z3 = np.dot(W3, a2) + b3
    a3 = g3(z3)

    return {'a1': a1, 'a2': a2, 'a3': a3}


def compute_gradient_three_layer(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3, g1_derivative, g2_derivative):
    propagation = forward_propagation_for_three_layered_neural_net(X, W1, b1, W2, b2, W3, b3, g1, g2, g3)

    a3 = propagation['a3']
    a2 = propagation['a2']
    a1 = propagation['a1']

    m = X.shape[1]

    dz3 = a3 - Y
    dw3 = (np.dot(dz3, a2.T))/m
    db3 = (np.sum(dz3, axis=1, keepdims=True))/m

    dz2 = np.dot(W3.T, dz3) * g1_derivative(a2)
    dw2 = (np.dot(dz2, X.T))/m
    db2 = (np.sum(dz2, axis=1, keepdims=True)) / m

    dz1 = np.dot(W2.T, dz2) * g2_derivative(a1)
    dw1 = (np.dot(dz1, X.T))/m
    db1 = (np.sum(dz1, axis=1, keepdims=True)) / m

    kost = cost(a3, Y)

    # W1 = W1 - learning_rate * dw1
    # b1 = b1 - learning_rate * db1
    # W2 = W2 - learning_rate * dw2
    # b2 = b2 - learning_rate * db2

    return {
        'cost': kost,
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2,
        'db2': db2,
        'dw3': dw3,
        'db3': db3,
    }
