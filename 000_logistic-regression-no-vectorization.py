import numpy as np

# logistic regression recap
#
# z = wT * x + b
# yHat = a = sigmoid(z
# loss(a, y) = -[y*log(a) + (1-y)*log(1-a)]

x1 = 2
w1 = 1
x2 = 3
w2 = 4
b = 5

z = x1 * w1 + x2 * w2 + b
print('z: ' + str(z))

X = np.array([x1, x2])
W = np.array([w1, w2])

print(W.dot(X) + b)
print(X.dot(W) + b)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


print(sigmoid(0))
print(sigmoid(5))
print(sigmoid(-5))


# a eh o yHat
# essa funcao nao funciona. tenho que tacar uns if para
# que ela funcione adequadamente, devido ao log(0)
def loss(a, y):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


print("loss: " + str(loss(.9999, 1)))
print("loss: " + str(loss(0.0001, 0)))


def forward_propagation(x1, x2, w1, w2, b):
    y = 0
    z = w1 * x1 + w2 * x2 + b
    a = sigmoid(z)
    return a


a = forward_propagation(x1, x2, w1, w2, b)
print('a: ' + str(a))
y = 1
dz = a - y
dw1 = x1 * dz
dw2 = x2 * dz
db = dz
print('dw1: ' + str(dw1))
print('dw2: ' + str(dw2))
print('db: ' + str(db))

# agora, vou fazer um gradient checking.
epsilon = .0001
w1_checking = (loss(forward_propagation(x1, x2, w1 + epsilon, w2, b), y) - loss(forward_propagation(x1, x2, w1, w2, b),
                                                                                y)) / epsilon
print('dw1 checking:' + str(w1_checking))

w2_checking = (loss(forward_propagation(x1, x2, w1, w2 + epsilon, b), y) - loss(forward_propagation(x1, x2, w1, w2, b),
                                                                                y)) / epsilon
print('dw2 checking:' + str(w2_checking))

db_checking = (loss(forward_propagation(x1, x2, w1, w2, b + epsilon), y) - loss(forward_propagation(x1, x2, w1, w2, b),
                                                                                y)) / epsilon
print('db checking:' + str(db_checking))

# my gradient checking is pumping fine!

# i'll try do it with vectorization now.
