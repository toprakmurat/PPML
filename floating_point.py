import numpy as np
from concrete import fhe
from time import time

from numpy.random import randint
from numpy.random import rand
from numpy import round


nb_bits = 3
length_inputset = 20
nb_test_samples = 8

max_value_for_floats = 2.5
scaling_factor = 6

def special_function_in_clear(x, y):
	u = x ** 2
	v = u + 4 * y
	return v / 1.33


@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def special_function(x, y):
	u = fhe.univariate(lambda x: x ** 2 // scaling_factor)(x)
	v = u + 4 * y
	return np.round(fhe.univariate(lambda x: x / 1.33)(v)).astype(np.int64)
