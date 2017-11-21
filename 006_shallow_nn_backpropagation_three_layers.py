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


W1, B1 = funcs.create_a_nn_layer(4, 2)
W2, B2 = funcs.create_a_nn_layer(3, 4)
W3, B3 = funcs.create_a_nn_layer(1, 3)

a1 = np.dot(W1, a0) + B1
a2 = np.dot(W2, a1) + B2
a3 = np.dot(W3, a2) + B3

back_prop = funcs.compute_gradient_three_layer(
    X=X, Y=Y,
    W1=W1, b1=B1,
    W2=W2, b2=B2,
    W3=W3, b3=B3,
    g1=funcs.sigmoid, g2=funcs.sigmoid, g3=funcs.sigmoid,
    g1_derivative=funcs.sigmoid_derivative, g2_derivative=funcs.sigmoid_derivative
)

# agora eh o seguinte: vou fazer uns gradient checking para ter certeza de que meus algoritmos
# de forward propagation e backpropagations estão bombando pra caralho.
# com isso, com meus testes bombando, depois, vou ser capaz de trocar minhas funcoes de ativacao
# e manter os testes funcionando.

funcs.do_gradient_checking(
    X=X, Y=Y,
    W1=W1, b1=B1,
    W2=W2, b2=B2,
    W3=W3, b3=B3,
    g1=funcs.relu, g2=funcs.relu, g3=funcs.sigmoid,
    g1_derivative=funcs.relu_derivative, g2_derivative=funcs.relu_derivative
)

print("just finished.")
