import numpy as np

a = np.array([[0.23875153, 0.58318603, 0.5816853,  0.19389093, 0.67800844], [0.20469368, 0.59248257, 0.38294148, 0.17688346, 0.89432704]])
b = np.array([[1, 0, 0, 0, 0], [0, 1, 0 , 0, 0]])
print(a - b)
c = (a - b)**2
print(c)
print(np.sum(c, axis=1) / 5)
print(np.mean(c, axis=1))
print(np.mean(np.sum(c, axis=1) / 5))