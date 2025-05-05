import numpy as np

def NormL1(v, dx):
	return dx*np.sum(np.abs(v[1:-1]))

def NormLInf(v, dx):
	return np.max(np.abs(v[1:-1]))

def NormL2(v, dx):
	v = v[1:-1]
	return np.sqrt(np.sum(dx * v**2))

def NormH1(v, dx):
	return np.sqrt(NormL2(v, dx)**2 + DxV_xhNorm2(v, dx))

def DxV_xhNorm2(v, dx):
	base = np.copy(v[1:])
	base = base - v[:-1]
	return np.sum(base**2)
