import numpy as np


def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return 1. * (relu(x) > 0)

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
    return np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a)))/Y_hat.shape[1]


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
    W = np.random.rand(number_of_neurons, number_of_features) * .001
    B = np.random.rand(number_of_neurons, 1) * .001

    return W, B


def forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3):
    a0 = X

    z1 = np.dot(W1, a0) + b1
    a1 = g1(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = g2(z2)

    z3 = np.dot(W3, a2) + b3
    a3 = g3(z3)

    kost = cost(a3, Y)

    return {'a1': a1, 'a2': a2, 'a3': a3, 'cost': kost}


def compute_gradient_three_layer(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3, g1_derivative, g2_derivative):
    propagation = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3)

    a3 = propagation['a3']
    a2 = propagation['a2']
    a1 = propagation['a1']
    a0 = X

    m = X.shape[1]

    dz3 = a3 - Y
    dw3 = (np.dot(dz3, a2.T))/m
    db3 = (np.sum(dz3, axis=1, keepdims=True))/m

    dz2 = np.dot(W3.T, dz3) * g2_derivative(a2)
    dw2 = (np.dot(dz2, a1.T))/m
    db2 = (np.sum(dz2, axis=1, keepdims=True)) / m

    dz1 = np.dot(W2.T, dz2) * g1_derivative(a1)
    dw1 = (np.dot(dz1, a0.T))/m
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



def do_gradient_checking(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3, g1_derivative, g2_derivative):
    epsilon = .000001

    gradient = compute_gradient_three_layer(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3, g1_derivative, g2_derivative)

    W1_1 = np.array(W1)
    W1_1[0][0] = W1_1[0][0] + epsilon
    propagation2 = forward_propagation_for_three_layered_neural_net(X, Y, W1_1, b1, W2, b2, W3, b3, g1, g2, g3)
    propagation1 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3)

    print((propagation2['cost'] - propagation1['cost']) / epsilon)
    print(gradient['dw1'][0][0])
    print('---')

    W1_2 = np.array(W1)
    W1_2[1][1] = W1_2[1][1] + epsilon
    propagation2 = forward_propagation_for_three_layered_neural_net(X, Y, W1_2, b1, W2, b2, W3, b3, g1, g2, g3)
    propagation1 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3)

    print((propagation2['cost'] - propagation1['cost']) / epsilon)
    print(gradient['dw1'][1][1])
    print('---')

    b1_1 = np.array(b1)
    b1_1[0][0] = b1_1[0][0] + epsilon
    propagation2 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1_1, W2, b2, W3, b3, g1, g2, g3)
    propagation1 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3)
    print((propagation2['cost'] - propagation1['cost']) / epsilon)
    print(gradient['db1'][0][0])
    print('---')

    W2_1 = np.array(W2)
    W2_1[1][2] = W2_1[1][2] + epsilon
    propagation2 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2_1, b2, W3, b3, g1, g2, g3)
    propagation1 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3)

    print((propagation2['cost'] - propagation1['cost']) / epsilon)
    print(gradient['dw2'][1][2])
    print('---')

    b2_1 = np.array(b2)
    b2_1[0][0] = b2_1[0][0] + epsilon
    propagation2 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2_1, W3, b3, g1, g2, g3)
    propagation1 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3)
    print((propagation2['cost'] - propagation1['cost']) / epsilon)
    print(gradient['db2'][0][0])
    print('---')

    W3_1 = np.array(W3)
    W3_1[0][0] = W3_1[0][0] + epsilon
    propagation2 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3_1, b3, g1, g2, g3)
    propagation1 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3)

    print((propagation2['cost'] - propagation1['cost']) / epsilon)
    print(gradient['dw3'][0][0])
    print('---')

    b3_1 = np.array(b3)
    b3_1[0][0] = b3_1[0][0] + epsilon
    propagation2 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3_1, g1, g2, g3)
    propagation1 = forward_propagation_for_three_layered_neural_net(X, Y, W1, b1, W2, b2, W3, b3, g1, g2, g3)
    print((propagation2['cost'] - propagation1['cost']) / epsilon)
    print(gradient['db3'][0][0])
    print('---')
