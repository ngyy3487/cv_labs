import numpy as np

a = [[1,2,3],[4,5,6]]
b = np.array(a)
print(np.transpose(b,[1,0])[::-1])