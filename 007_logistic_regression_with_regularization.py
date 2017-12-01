import numpy as np


# passo 1 -> fazer minha forward propagation vetorizada
# passo 2 -> fazer meu gradiente vetorizado
# passo 3 -> fazer o passo 1 com tensorflow
# passo 4 -> fazer o passo 2 com tensorflow
# passo 5 -> fazer o gradient checking com numpy
# passo 6 -> fazer o gradient checking com tensorflow
# passo 7 -> introduzir regularização com numpy
# passo 8 -> introduzir regularização com tensorflow


if __name__ == '__main__':
    point1 = np.array([1, 2])
    point2 = np.array([3, 5])
    point3 = np.array([4, 7])

    X = np.array([point1, point2, point3]).T
    Y = np.array([[1, 1, 0]])

    assert(X.shape == (2, 3))
    assert(Y.shape == (1, 3))

    # passo 1 -> fazer minha forward propagation vetorizada

    print("just finished")
