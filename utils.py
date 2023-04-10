import numpy as np

def generate_normal(size=100, num_k=3):
	vals = []
	num_c = 100

	d = np.array([[2. * j + 1 for j in range(1, num_c + 1)]])
	ln = np.array([[1. / (2 * num_k) * np.log(1 + 1. / j) for j in range(1, num_c + 1)]])
	
	s1 = np.random.gamma(1. / (num_k), 1, [size, num_c])
	r = 2 * ((np.random.uniform(0., 1., size) > 0.5).astype(np.float) - 0.5)
	# print (s1.shape, r.shape, (s1 / d - ln).sum(axis = -1).shape, np.exp(np.log(2) / 4 - np.random.gamma(0.5, 1, [size]) - (s1 / d - ln).sum(axis = -1)).shape)
	v1 = r * np.exp(np.log(2) / (2. * num_k) - np.random.gamma(1. / num_k, 1, [size]) - (s1 / d - ln).sum(axis = -1))

	# s2 = np.random.gamma(0.5, 1, [size, num_c])
	# r = 2 * ((np.random.uniform(0., 1., size) > 0.5).astype(np.float) - 0.5)
	# v2 = r * np.exp(np.log(2) / 4 - np.random.gamma(0.5, 1, [size]) - (s2 / d - ln).sum(axis = -1))
	
	# vs += v1 * v2

	return v1