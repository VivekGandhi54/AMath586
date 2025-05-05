import numpy as np
import matplotlib.pyplot as plt

# For A
from scipy.sparse import spdiags
# To solve AL = U
from scipy.sparse.linalg import gmres, spsolve
# from scipy.linalg import solve
from Norms import *

# ----------------------------------------------

do_viz = False

# ----------------------------------------------
# Explicit solution

def v(t, x):
	coef = 1/np.sqrt(4*np.pi*t)
	exp = np.exp(-((x-2)**2)/(4*t))
	return coef*exp

# ==============================================

def main():
	Ns = [20, 40, 80, 160]
	dxs = [10/(n+1) for n in Ns]
	modes = ['EE', 'CN', 'IE']

	for mode in modes:
		L1s = []
		L2s = []
		Linfs = []
		H1s = []

		for n in Ns:
			sol, error = theta_scheme(mode, n)
			viz(sol, error, f'{mode} (n = {n})')

			eT = error[0]	# Top slice - error at time T
			dx = 10/(n+1)

			L1s.append(NormL1(eT, dx))
			L2s.append(NormL2(eT, dx))
			Linfs.append(NormLInf(eT, dx))
			H1s.append(NormH1(eT, dx))

		get_slope = lambda errors: np.polyfit(np.log10(dxs), np.log10(errors), 1)[0]

		plt.loglog(dxs, L1s, label='L1 Norm', linestyle='--', marker='o')
		plt.loglog(dxs, L2s, label='L2 Norm', linestyle='--', marker='o')
		plt.loglog(dxs, Linfs, label='L_inf Norm', linestyle='--', marker='o')
		plt.loglog(dxs, H1s, label='H1 Norm', linestyle='--', marker='o')
		plt.legend()
		plt.title(f'{mode}')
		plt.xlabel('h = 1/(N + 1) (log)')
		plt.ylabel('Error Norm (log)')
		plt.tight_layout()
		plt.show()

		print(f'\n{mode}')
		print(f'L1 slope - {get_slope(L1s)}')
		print(f'L2 slope - {get_slope(L2s)}')
		print(f'L_inf slope - {get_slope(Linfs)}')
		print(f'H1 slope - {get_slope(H1s)}')

# ==============================================

def theta_scheme(mode, n):
	T = 3

	# ------ Setup -----------------------

	xRange = np.linspace(-5, 5, n+2)
	dx = np.diff(xRange)[0]
	dt = 0.4*(dx**2) if mode == 'EE' else dx
	dt_dx2 = dt/(dx*dx)

	tRange = np.arange(0, T+dt, dt)

	theta = 1		# IE
	if mode == 'EE':
		theta = 0
	elif mode == 'CN':
		theta = 0.5

	# ------ Produce u0 ------------------

	un = generate_u0(xRange, lambda x: v(1, x))
	u = np.array([un])

	# ------ Produce ImpMat and ExpMat ---

	A = 1 + (2*theta*dt_dx2)		# Implicit diagonal
	B = -theta*dt_dx2				# Implicit off-diagonal
	C = 1 - (1 - theta)*2*dt_dx2	# Explicit diagonal
	D = (1 - theta)*dt_dx2			# Explicit off-diagonal

	ImpMat = buildMatrix(n, A, B)
	ExpMat = buildMatrix(n+2, C, D)		# n+2 to include U_0 and U_N+1

	# ------ Step through time -----------

	for t in tRange[1:]:
		F = (ExpMat @ un)[1:-1]			# Calculate explicit part

		U_0 = v(t+1, -5)
		U_N1 = v(t+1, 5)

		if mode == 'EE':
			un = F						# Return early if fully explicit
		else:
			F[0] -= B*U_0
			F[-1] -= B*U_N1
			un = spsolve(ImpMat, F)		# Solve implicit part

		un = np.concatenate(([U_0], un, [U_N1]))
		u = np.insert(u, 0, [un], axis=0)

	# ------ Calculate error -------------

	real = generate_full(xRange, tRange, lambda t, x: v(t+1, x))
	error = real-u

	return [u, error]

# ==============================================
# Helpers
# ==============================================

# Generate an nxn matrix across ranges with an initializer func
def generate_u0(xRange, func):
	w0 = np.zeros(xRange.size)
	for x in range(xRange.size):
		w0[x] = func(xRange[x])
	return w0

# ----------------------------------------------
# Generate an nxn matrix across ranges with an initializer func
def generate_full(xRange, tRange, func):
	w0 = np.zeros((tRange.size, xRange.size))
	for x in range(xRange.size):
		for t in range(tRange.size):
			w0[t,x] = func(tRange[t], xRange[x])
	return w0[::-1]

# ----------------------------------------------
# Vizualize U and error=U-u
def viz(A, B, title=''):
	if not do_viz:
		return
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
	im1 = ax1.imshow(A, cmap='jet')
	im2 = ax2.imshow(B, cmap='jet')
	ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
	ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
	ax1.set_box_aspect(1)
	ax2.set_box_aspect(1)
	ax1.set_title(f'Solution --- {title}')
	ax2.set_title(f'Error')

	fig.colorbar(im1, ax=ax1)
	fig.colorbar(im2, ax=ax2)

	plt.tight_layout()
	plt.show()

# ==============================================
# Builds an nxn matrix with A on the diagonal and
# 	B on the off-diagonals
# ==============================================

def buildMatrix(n, A, B):
	e1 = np.ones(n)			# vector of 1s

	diagonals = [B*e1, A*e1, B*e1]
	offsets = [-1, 0, 1]

	mat = spdiags(diagonals, offsets, n, n, format = 'csr')
	return mat

# ==============================================

main()
