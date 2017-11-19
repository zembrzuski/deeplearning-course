import general_functions as funcs
import numpy as np

# minha rede neural vai ter uma única layer, por enquanto.


# se minha input layer tem shape==(2,3) pois possui duas features
# e tem 3 elementos, entao minha layer do meio vai ter shape==(4,3)
# pois minha layer do meio tem 4 neuronios, que equivalem a features
# e 3 training examples

x1 = np.array([1, -1])
x2 = np.array([2, 0])
x3 = np.array([-2, 3])

X = np.array([x1, x2, x3]).T
a0 = X
assert (a0.shape == (2, 3))  # 2 features e 3 m training examples.

w11 = np.array([2, 100]).reshape((1, 2))
b11 = 1

z11 = np.dot(w11, X) + b11
print(z11)

w12 = np.array([4, 200]).reshape((1, 2))
b12 = 8

z12 = np.dot(w12, X) + b12
print(z12)

w13 = np.array([5, -1]).reshape((1, 2))
b13 = 4
z13 = np.dot(w13, X) + b13
print(z13)

w14 = np.array([1, -2]).reshape((1, 2))
b14 = 2
z14 = np.dot(w14, X) + b14
print(z14)

W1 = np.concatenate([w11, w12, w13, w14])
b1 = np.array([b11, b12, b13, b14]).reshape((4, 1))
assert (W1.shape == (4, 2))  # 4 neuronios, 2 features.

z1 = np.dot(W1, a0) + b1
assert (z1.shape == (4, 3))
print(z1)
a1 = funcs.sigmoid(z1)

w21 = np.array([1, -1, 2, -2]).reshape((1, 4))  # um neuronio, 4 features.
W2 = np.concatenate([w21])
b2 = np.array([-.1]).reshape((1, 1))

z2 = np.dot(W2, a1) + b2
assert(z2.shape == (1, 3))
a2 = funcs.sigmoid(z2)

print(a2)