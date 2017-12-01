import numpy as np

padding = 1

ponto1_red = np.array([
    ['1R', '2R'],
    ['3R', '4R']
])

ponto1_green = np.array([
    ['1G', '2G'],
    ['3G', '4G']
])

ponto1_blue = np.array([
    ['1B', '2B'],
    ['3B', '4B']
])

ponto1 = np.array([ponto1_red, ponto1_green, ponto1_blue]).T

ponto2_red = np.array([
    ['5R', '6R'],
    ['7R', '5R']
])

ponto2_green = np.array([
    ['5G', '6G'],
    ['7G', '5G']
])

ponto2_blue = np.array([
    ['5B', '6B'],
    ['7B', '5B']
])


ponto2 = np.array([ponto2_red, ponto2_green, ponto2_blue]).T


X = np.array([ponto1, ponto2])

print(X.shape)
print('----')
simple_matrix_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
print(simple_matrix_padded[0][1])
