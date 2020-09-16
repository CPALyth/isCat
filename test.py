import numpy as np

lst = [1,2,3,4,5,6]
a1 = np.array(lst)
print(a1)
a2 = a1.reshape((6,-1))
print(a2)
a3 = np.squeeze(a2)
print(a3)