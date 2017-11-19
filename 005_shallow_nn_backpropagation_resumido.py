import general_functions as funcs
import numpy as np


# criei minha input layer.
x1 = np.array([1., -1])
x2 = np.array([2.,  0])
x3 = np.array([-2., 3])
X = np.array([x1, x2, x3]).T
a0 = X
assert (a0.shape == (2, 3))  # 2 features e 3 m training examples.

Y = np.array([[1., 1, 1]])
assert(Y.shape == (1, 3))

# vou criar minha primeira hidden layer agora.

w11 = np.array([2., 100]).reshape((1, 2))
b11 = 1.
w12 = np.array([4., 200]).reshape((1, 2))
b12 = 8.
w13 = np.array([5., -1]).reshape((1, 2))
b13 = 4
w14 = np.array([1., -2]).reshape((1, 2))
b14 = 2.

W1 = np.concatenate([w11, w12, w13, w14])
b1 = np.array([b11, b12, b13, b14]).reshape((4, 1))
w21 = np.array([1., -1, 2, -2]).reshape((1, 4))  # um neuronio, 4 features.
W2 = np.concatenate([w21])
b2 = np.array([-.1]).reshape((1, 1))

m = 3 # 3 training examples
n0 = 2   # 2 features
n1 = 4 # 4 neuronios (equivalente a features)
n2 = 1 # 1 neuronio.

assert(W1.shape == (n1, n0))
assert(b1.shape == (n1, 1))
assert(W2.shape == (n2, n1))
assert(b2.shape == (n2, 1))


gradient_computed = funcs.compute_gradient(X, Y, W1, b1, W2, b2, .1)

# acredito que isso aqui esta funcionando ate agora.
# o proximo passo eh fazer meu gradient checking.

epsilon = .0001
b2_modified = np.array(b2)
b2_modified[0][0] = b2_modified[0][0] + epsilon


print((funcs.compute_gradient(X, Y, W1, b1, W2, b2_modified, .1)['cost'] - funcs.compute_gradient(X, Y, W1, b1, W2, b2, .1)['cost']) / epsilon)

xoxo = funcs.compute_gradient(X, Y, W1, b1, W2, b2, .1)
print('oi')

# TODO passo 1: fazer gradient checking com cada uma das variaveis para ver se esse cara funciona.
# TODO passo 2: trocar funcao de ativacao para ReLU e ver que ainda funciona.
# TODO E eh isso.



print('finished')
