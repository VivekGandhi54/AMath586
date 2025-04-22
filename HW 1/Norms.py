import numpy as np

def NormL1(v, h):
	return (h**2)*np.sum(np.abs(v[1:-1, 1:-1]))

def NormLInf(v, h):
	return np.max(np.abs(v[1:-1, 1:-1]))

def NormL2(v, h):
	v = v[1:-1, 1:-1]
	return h * np.sqrt(np.sum(v**2))

def NormH1(v, h):
	return np.sqrt(NormL2(v, h)**2 + DxV_xhNorm2(v, h) + DyV_yhNorm2(v, h))

def DxV_xhNorm2(v, h):
	base = np.copy(v[1:-1, 1:])
	base = base - v[1:-1, :-1]
	return np.sum(base**2)

def DyV_yhNorm2(v, h):
	return DxV_xhNorm2(v.transpose(), h)
