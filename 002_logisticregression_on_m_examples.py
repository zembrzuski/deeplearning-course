import numpy as np


def loss(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_gradient(X, Y, W, b):
    J = 0; dw1 = 0; dw2 = 0; db = 0
    m = X.shape[1]

    for i in range(0, m):
        xi = X[:,i]
        yi = Y[i]
        zi = W.dot(xi) + b
        ai = sigmoid(zi)
        J += loss(ai, yi)
        dzi = ai - yi
        dw1 += xi[0] * dzi
        dw2 += xi[1] * dzi
        db += dzi

    J /= m; dw1 /=m; dw2 /=m; db /= m

    return {'cost': J, 'weight derivative': np.array([dw1, dw2]), 'bias derivative': db}


if __name__ == '__main__':
    point1 = np.array([1, 2])
    point2 = np.array([3, 5])
    point3 = np.array([4, 7])

    X = np.array([point1, point2, point3]).T
    Y = np.array([1, 1, 0])
    W = np.array([.5, -.5])
    b = .3

    gradient = compute_gradient(X, Y, W, b)
    print(gradient)

    # TODO fazer um gradient checking para ver se esse cara funciona direito.

    # TODO fazer isso com vetorization.

    # TODO rever o grafo de deep learning para entender direitinho o algoritmo