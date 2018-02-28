import numpy as np
import math

def hanning(size):
	v = np.divide(np.arange(size-1)/(size-1))
	arr_size = len(v)
	v2 = np.ones(arr_size)

	v2[0: arr_size] = v
	hanv = - np.cos(v2 * 2 * math.pi) * 0.5 + 0.5
	ret = np.add(np.out(hanv, hanv), 0.01)
	ret = np.divide(ret, ret.sum())

	return ret
