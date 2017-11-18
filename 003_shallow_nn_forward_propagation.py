import numpy as np
import general_functions as funcs

# vou implementar o forward propagation de uma rede neural.

#
# criando o input layer
#

x1 = np.array([1, -1])
x2 = np.array([2, 0])
x3 = np.array([-2, 3])

X = np.array([x1, x2, x3]).T
a0 = X
assert (a0.shape == (2, 3))  # 2 features e 3 m training examples.


#
# criando minha hidden layer (no primeiro momento, com somente 1 neuronio.
# depois eu vou colocando outros neuronios e generalizando para n neuronios.
#

# meu primeiro neuronio
w11 = np.array([2, 100]).reshape((1, 2))
b11 = 1
w12 = np.array([4, 200]).reshape((1, 2))
b12 = 8
w13 = np.array([5, -1]).reshape((1, 2))
b13 = 4
w14 = np.array([1, -2]).reshape((1, 2))
b14 = 2

W1 = np.concatenate([w11, w12, w13, w14])
b1 = np.array([b11, b12, b13, b14]).reshape((4, 1))
assert(W1.shape == (4, 2)) # 4 neuronios, 2 features.

a1 = funcs.sigmoid(np.dot(W1, a0) + b1)
assert (a1.shape == (4, 3))  # 4 features (neuronios) e 3 m training examples.

w21 = np.array([1, -1, 2, -2]).reshape((1, 4))  # um neuronio, 4 features.
W2 = np.concatenate([w21])
b2 = np.array([-.1]).reshape((1, 1))
assert (W2.shape == (1, 4))

a2 = funcs.sigmoid(np.dot(W2, a1) + b2)
assert(a2.shape == (1, 3))
print(a2)
print('deu')
