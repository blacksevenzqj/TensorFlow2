import numpy as np


a = np.array([[1,11,111,1111],[2,22,222,2222],[3,33,333,3333]])
b = np.array([[6,66],[7,77],[8,88],[9,99]])

print(a.dot(b))
print(b.T.dot(a.T))