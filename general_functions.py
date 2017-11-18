import numpy as np


def sigmoid(x):
    """
    Computa a sigmoide de x. X pode ser um número real ou um numpy.array
    """
    return 1 / (1 + np.exp(-x))
