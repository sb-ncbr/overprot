import numpy as np
import random

def antidiagonal(X: np.ndarray, d: int):
	'''Extracts d-th antidiagonal from a matrix X (0th antidiagonal is X[0,0], the last is X[-1,-1]).'''
	m, n = X.shape
	Y = X.reshape(-1)
	if d > m + n - 2:
		raise IndexError()
	if d < n:
		start = d
	else:
		start = n*d - (n-1)*(n-1)
	if d < m:
		end = n*d + 1
	else:
		end = m*n - m - n + 2 + d
	return Y[start:end:n-1]

def test():
	for i in range(1000):
		m = random.randint(1, 2000)
		n = random.randint(1, 2000)
		print(m, n)
		X = np.arange(m*n).reshape((m, n))
		seen = set()
		for d in range(m+n-1):
			ad = antidiagonal(X, d)
			for x in ad:
				if x in seen:
					raise Exception()
				else:
					seen.add(x)
		if len(seen) != m*n:
			raise Exception()
		try:
			antidiagonal(X, m+n-1)
			raise Exception()
		except IndexError:
			pass

test()
exit(0)

m, n = 5, 8
X = np.arange(m*n).reshape((m, n))

print(X)
for i in range(m+n):
    print(i)
    print(antidiagonal(X, i))
