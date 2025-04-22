
import numpy as np
import matplotlib.pyplot as plt

# For A
from scipy.sparse import spdiags
# To solve AL = U
from scipy.sparse.linalg import gmres, spsolve
from scipy.linalg import solve
from Norms import *

# ----------------------------------------------

do_matrix_viz = False	# True to see matrix A's structure
do_viz = False			# True to plot the solution and error
mat_viz_mode = 0

# ==============================================

def main():
	# Ns = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
	Ns = [20, 40, 80, 160]
	Hs = [1/(n+1) for n in Ns]

	L1s = []
	L2s = []
	Linfs = []
	H1s = []

	for n in Ns:
		data = gather_norms(n)

		L1s.append(data[0])
		L2s.append(data[1])
		Linfs.append(data[2])
		H1s.append(data[3])

	get_slope = lambda errors: np.polyfit(np.log10(Hs), np.log10(errors), 1)[0]

	plt.loglog(Hs, L1s, label='L1 Norm', linestyle='--', marker='o')
	plt.loglog(Hs, L2s, label='L2 Norm', linestyle='--', marker='o')
	plt.loglog(Hs, Linfs, label='L_inf Norm', linestyle='--', marker='o')
	plt.loglog(Hs, H1s, label='H1 Norm', linestyle='--', marker='o')
	plt.legend()
	plt.xlabel('h = 1/(N + 1) (log)')
	plt.ylabel('Error Norm (log)')
	plt.show()

	print(f'L1 slope - {get_slope(L1s)}')
	print(f'L2 slope - {get_slope(L2s)}')
	print(f'L_inf slope - {get_slope(Linfs)}')
	print(f'H1 slope - {get_slope(H1s)}')

# ==============================================

def gather_norms(n = 200):

	h = 1/(n+1)
	h2_n = (n+1)**2
	
	# ------ Setup -----------------------
	
	xRange = np.linspace(0, 1, n+2)
	yRange = np.linspace(0, 1, n+2)

	# ------ Produce A -------------------

	A = buildA(n, h2_n)
	vizMatrix(A)

	# ------ Produce F -------------------

	BC_0 = np.sin(np.pi*xRange[1:-1])
	BC_1 = -np.sin(np.pi*xRange[1:-1])
	BC = np.concatenate((BC_0, np.zeros(n*(n-2)), BC_1))

	initializer = lambda y, x: 2*(np.pi**2)*np.sin(np.pi*x)*np.cos(np.pi*y)
	w0 = generate_w0(xRange[1:-1], yRange[1:-1], initializer)

	F = deform(w0) + h2_n*deform(BC)

	# ------ Solve for U -----------------

	U_sol = spsolve(A, F)				# Sparse solve
	# U_sol = solve(A.toarray(), F)		# Dense solve
	# U_sol = gmres(A, F)[0]			# GMREs

	U_sol = reform(U_sol)

	# ------ Append BCs ------------------

	U = np.zeros((n+2, n+2))
	U[1:1+U_sol.shape[0], 1:1+U_sol.shape[1]] = U_sol
	U[0,:] = np.sin(np.pi*xRange)
	U[-1,:] = -np.sin(np.pi*xRange)

	viz(U, f'Solution (n={n})')

	# ------ Compare against solution-----

	initializer = lambda y, x: np.sin(np.pi*x)*np.cos(np.pi*y)
	real = generate_w0(xRange, yRange, initializer)

	error = real-U
	viz(error, f'Error (n={n})')

	return [NormL1(error, h), NormL2(error, h),
				NormLInf(error, h), NormH1(error, h)]

# ==============================================
# Helpers
# ==============================================

# Unroll an nxn matrix into an mx1 one
def deform(A):
	return A.reshape((A.size, 1))

# ----------------------------------------------
# Reform an mx1 matrix into an nxn one
def reform(A):
	n = np.sqrt(A.size).astype(int)
	return A.reshape((n, n))

# ----------------------------------------------
# Generate an nxn matrix across ranges with an initializer func
def generate_w0(xRange, yRange, func):
	w0 = np.zeros((yRange.size, xRange.size))
	for x in range(xRange.size):
		for y in range(yRange.size):
			w0[x,y] = func(xRange[x], yRange[y])
	return w0

# ----------------------------------------------
# Vizualize U or error=U-u
def viz(A, title):
	if do_viz:
		plt.imshow(A, cmap='jet')
		plt.colorbar()
		plt.title(title)
		plt.show()

# ----------------------------------------------
# Visualize matrix A using either [0] plt.spy() or [1] plt.imshow()
def vizMatrix(mat):
	global mat_viz_mode
	if not do_matrix_viz:
		return
	if mat_viz_mode == 0:
		plt.figure(5)
		plt.spy(mat)
	else:
		plt.imshow(mat.toarray(), interpolation='none', cmap='binary')
		plt.colorbar()
	plt.title('Matrix Structure')
	plt.show()

# ==============================================
# A = -(dxx + dyy)
# ==============================================

def buildA(n, h2_n):
	m = n**2				# Size of A

	e1 = np.ones(m)			# vector of 1s
	e4 = 4*np.copy(e1)		# vector of 4s

	e_n = np.ones(n)
	e_n[-1] = 0
	e_n = -1*np.tile(e_n, n)	# vector of -1s, with 0 at the end

	e_m = np.roll(e_n, 1)

	diagonals = [-1*e1, e_n, e4, e_m, -1*e1]
	offsets = [-n, -1, 0, 1, n]

	matA = spdiags(diagonals, offsets, m, m, format = 'csr')
	return h2_n * matA

# ==============================================

main()
