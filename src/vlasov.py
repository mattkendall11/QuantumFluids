import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import math
import numpy.linalg as LA
from scipy.sparse.linalg import LinearOperator

# -------------------------
# Constants and Parameters
# -------------------------
e = 1.6e-19
m_e = 9.1e-31
eps_0 = 8.85e-12
k_B = 1.38e-23
T = 8000.0
b = m_e / (2 * k_B * T)
v_p = 1 / np.sqrt(b)

xmax = 1e6
vmax = 10.0 * v_p

Nx, Nv = 2, 2
N = Nx * Nv
N_particles = 5e5 * 2 * xmax
m = 1  # Carleman truncation

# -------------------------
# Helper functions
# -------------------------
def ddslash(a_1, a_2):
    return a_1 - a_2 * (ceil(a_1/a_2) - 1)

def kron_delta(a, b):
    return 1 if int(a) == int(b) else 0

def vj(j_phys, Nv):
    d_v = 2 * vmax / (Nv - 1)
    return -vmax + (j_phys - 1) * d_v

def nuj(j_phys, Nv, p, n, nu_0):
    return nu_0 + p * (np.abs(vj(j_phys, Nv)) / vmax)**n

def lognorm(A):
    sym_matrix = 0.5 * (A + A.T)
    eigenvalues, _ = LA.eig(sym_matrix)
    return np.max(eigenvalues)

# -------------------------
# Initial condition
# -------------------------
def fill_matrix(Nx, Nv, J):
    matrix = np.zeros((Nx, Nv))
    d_v = 2 * vmax / (Nv - 1)
    for i in range(Nx):
        for j in range(Nv):
            if j == J or j == Nv - J - 1:
                matrix[i, j] = N_particles / (2.0 * xmax * d_v)
    return matrix

def u_n(f, Nx, Nv):
    N = Nx * Nv
    un = np.zeros(N)
    for n in range(N):
        n_phys = n + 1
        un[n] = f[ceil(n_phys/Nv) - 1, ddslash(n_phys, Nv) - 1]
    return un

def U_tensor_U(U):
    N = len(U)
    U_tensor_U = np.zeros(N**2)
    for n in range(N**2):
        n_phys = n + 1
        U_tensor_U[n] = U[ceil(n_phys/N)-1] * U[ddslash(n_phys, N) - 1]
    return U_tensor_U

def create_F0(Nx,Nv,p,n,nu_0):
    d_x = xmax / Nx
    d_v = 2 * vmax / (Nv-1)
    N = Nx * Nv

    F0 = np.zeros(N)
    normalization_summation = 0.0

    for j in np.arange(1,Nv+1):
        normalization_summation+=np.exp(-b * vj(j,Nv)**2)

    M = N_particles / (2.0 * xmax * d_v * normalization_summation)
    #print('sum in M exact',normalization_summation)

    for n in np.arange(0,N):
        n_phys = n + 1
        F0[n] = M * np.exp(-b * vj(ddslash(n_phys, Nv),Nv)**2) * nuj(ddslash(n_phys, Nv),Nv,p,n,nu_0)

    return F0

def create_F1_a(Nx, Nv):
    N = Nx * Nv
    d_x = xmax / Nx
    d_v = 2 * vmax / (Nv - 1)
    F1 = np.zeros((N, N))
    for n in range(N):
        n_phys = n + 1
        v_term = -vj(ddslash(n_phys, Nv), Nv) / (2 * d_x)
        for k in range(N):
            k_phys = k + 1
            if n_phys <= Nv:
                F1[n, k] = v_term * (kron_delta(k_phys, n_phys + Nv) - kron_delta(k_phys, n_phys + Nv*(Nx-1)))
            elif Nv < n_phys <= Nv*(Nx-1):
                F1[n, k] = v_term * (kron_delta(k_phys, n_phys + Nv) - kron_delta(k_phys, n_phys - Nv))
            else:
                F1[n, k] = v_term * (kron_delta(k_phys, n_phys - Nv*(Nx-1)) - kron_delta(k_phys, n_phys - Nv))
    for n in range(N):
        n_phys = n + 1
        gauss_term = (e**2 * N_particles) / (2 * m_e * eps_0 * d_v) * (ceil(n_phys/Nv) - 1) / Nx
        for k in range(N):
            k_phys = k + 1
            if ddslash(n_phys, Nv) == 1:
                F1[n, k] += gauss_term * kron_delta(k_phys, n_phys + 1)
            elif ddslash(n_phys, Nv) == Nv:
                F1[n, k] -= gauss_term * kron_delta(k_phys, n_phys - 1)
            else:
                F1[n, k] += gauss_term * (kron_delta(k_phys, n_phys + 1) - kron_delta(k_phys, n_phys - 1))
    return F1

def create_F1_b(Nx, Nv, p, n, nu_0):
    N = Nx * Nv
    F1b = np.zeros((N, N))
    for n in range(N):
        n_phys = n + 1
        for k in range(N):
            k_phys = k + 1
            F1b[n, k] = -kron_delta(k_phys, n_phys) * nuj(ddslash(n_phys, Nv), Nv, p, n, nu_0)
    return F1b

def F2_n_phys_k_phys(Nx, Nv,n_phys,k_phys):
    N  = Nv * Nx
    delta_x = xmax / Nx

    K_const = -e**2 *delta_x / (4 * m_e *eps_0)
    ceiling = ceil(n_phys/Nv)

    if n_phys < Nv+1:
        return 0

    elif n_phys == Nv + 1:
        A = kron_delta(k_phys,N * n_phys + 1) + kron_delta(k_phys, N * n_phys + Nv + 1)
        A+= kron_delta(k_phys,N * n_phys + Nv) + kron_delta(k_phys,N * n_phys + 2 * Nv)

        for J in np.arange(2, Nv):
            A+= 2 * ( kron_delta(k_phys,N * n_phys + J) + kron_delta(k_phys,N * n_phys + Nv + J) )

        return K_const * A

    elif Nv + 1 < n_phys < 2 * Nv:
        recursive_1 = F2_n_phys_k_phys(Nx, Nv,Nv+1,k_phys + N * (Nv + 1 - n_phys))
        recursive_2 = F2_n_phys_k_phys(Nx, Nv,Nv+1,k_phys + N * (Nv + 3 - n_phys))

        return  recursive_1 - recursive_2

    elif n_phys == 2 * Nv:
        recursive_3 = F2_n_phys_k_phys(Nx, Nv,Nv+1,k_phys - N * (Nv - 3))

        return -1*recursive_3

    elif ceiling >= 3 and ddslash(n_phys,Nv)==1:

        C = kron_delta(k_phys,N * n_phys + 1) + kron_delta(k_phys,N * n_phys + Nv * (ceiling-1) + 1)
        C+= kron_delta(k_phys,N * n_phys + Nv) + kron_delta(k_phys,N * n_phys + ceiling * Nv)

        for I in np.arange(2, ceiling):
            C+= 2 * ( kron_delta(k_phys,N * n_phys + (I-1) * Nv + 1) + kron_delta(k_phys,N * n_phys + I * Nv) )

        for J in np.arange(2, Nv):
            C+= 2 * ( kron_delta(k_phys,N * n_phys + J) + kron_delta(k_phys,N * n_phys + Nv * (ceiling-1) + J) )

        for I in np.arange(2, ceiling):
            for J in np.arange(2, Nv):
                C+= 4 * kron_delta(k_phys,N * n_phys + (I-1) * Nv+ J)

        return 1.0 * K_const * C

    elif ceiling >= 3 and ddslash(n_phys,Nv)==Nv:
        D = kron_delta(k_phys,N * (n_phys-2) + 1) + kron_delta(k_phys,N * (n_phys-2) + Nv * (ceiling-1) + 1)
        D+= kron_delta(k_phys,N * (n_phys-2) + Nv) + kron_delta(k_phys,N * (n_phys-2) + ceiling * Nv)

        for I in np.arange(2, ceiling):
            D+= 2 * ( kron_delta(k_phys,N * (n_phys-2) + (I-1) * Nv + 1) + kron_delta(k_phys,N * (n_phys-2) + I * Nv) )

        for J in np.arange(2, Nv):
            D+= 2 * ( kron_delta(k_phys,N * (n_phys-2) + J) + kron_delta(k_phys,N * (n_phys-2) + Nv * (ceiling-1) + J) )

        for I in np.arange(2, ceiling):
            for J in np.arange(2, Nv):
                D+= 4 * kron_delta(k_phys,N * (n_phys-2) + (I-1) * Nv+ J)

        return -1.0 * K_const * D

    else:
        E_pos = kron_delta(k_phys,N * n_phys + 1) + kron_delta(k_phys,N * n_phys + Nv * (ceiling-1) + 1)
        E_pos+= kron_delta(k_phys,N * n_phys + Nv) + kron_delta(k_phys,N * n_phys + ceiling * Nv)

        for I in np.arange(2, ceiling):
            E_pos+= 2 * ( kron_delta(k_phys,N * n_phys + (I-1) * Nv + 1) + kron_delta(k_phys,N * n_phys + I * Nv) )

        for J in np.arange(2, Nv):
            E_pos+= 2 * ( kron_delta(k_phys,N * n_phys + J) + kron_delta(k_phys,N * n_phys + Nv * (ceiling-1) + J) )

        for I in np.arange(2, ceiling):
            for J in np.arange(2, Nv):
                E_pos+= 4 * kron_delta(k_phys,N * n_phys + (I-1) * Nv+ J)

        E_neg = kron_delta(k_phys,N * (n_phys-2) + 1) + kron_delta(k_phys,N * (n_phys-2) + Nv * (ceiling-1) + 1)
        E_neg+= kron_delta(k_phys,N * (n_phys-2) + Nv) + kron_delta(k_phys,N * (n_phys-2) + ceiling * Nv)

        for I in np.arange(2, ceiling):
            E_neg+= 2 * ( kron_delta(k_phys,N * (n_phys-2) + (I-1) * Nv + 1) + kron_delta(k_phys,N * (n_phys-2) + I * Nv) )

        for J in np.arange(2, Nv):
            E_neg+= 2 * ( kron_delta(k_phys,N * (n_phys-2) + J) + kron_delta(k_phys,N * (n_phys-2) + Nv * (ceiling-1) + J) )

        for I in np.arange(2, ceiling):
            for J in np.arange(2, Nv):
                E_neg+= 4 * kron_delta(k_phys,N * (n_phys-2) + (I-1) * Nv+ J)

        return 1.0 * K_const * (E_pos - E_neg)



def create_F2(Nx,Nv):
    N  = Nv * Nx
    F2 = np.zeros((N,N**2))

    for n in range(0, N):
        n_phys = n + 1
        for k in range(0, N**2):
            k_phys = k + 1
            F2[n,k] = F2_n_phys_k_phys(Nx, Nv ,n_phys,k_phys)

    return F2
# Full implementation would follow the recursive logic defined in original shared function
F2 = create_F2(Nx, Nv)

# -------------------------
# Carleman Embedding
# -------------------------
def construct_A_operator(F0, F1, F2, r):
    size = r * N
    def apply_A(v):
        out = np.zeros_like(v, dtype=np.float64)
        for j in range(r):
            base_idx = j * N
            block = v[base_idx:base_idx + N]
            if j < r:
                out[base_idx:base_idx + N] += np.float64(j) * np.array(F1, dtype = 'float64') @ np.array(block, dtype='float64')
            if j > 0:
                prev_block = v[(j - 1) * N : j * N]
                out[base_idx:base_idx + N] += j * F0 * prev_block
            if j < r - 1:
                next_block = v[(j + 1) * N : (j + 2) * N]
                out[base_idx:base_idx + N] += (j + 1) * (F2 @ U_tensor_U(next_block))
        return out
    return LinearOperator((size, size), matvec=apply_A)

# -------------------------
# Build everything
# -------------------------
J = Nv // 4
f = fill_matrix(Nx, Nv, J)
f_vec = u_n(f, Nx, Nv)
p, n, nu_0 = 0.0, 2.0, 0.0

F0 = create_F0(Nx, Nv, p, n, nu_0)
F1 = create_F1_a(Nx, Nv) + create_F1_b(Nx, Nv, p, n, nu_0)
F2 = np.zeros((N, N**2))  # Replace with full create_F2(Nx, Nv) for nonlinear dynamics

gamma = lognorm(np.diag(F0))  # Scalar rescaling
if gamma ==0:
    gamma = 1e-15

F0_bar = F0 / gamma
F1_bar = F1
F2_bar = F2 / gamma**2

A_op = construct_A_operator(F0_bar, F1_bar, F2_bar, m)

z0 = np.zeros(m * N)
z0[:N] = f_vec / gamma
b = np.zeros_like(z0)
psi = z0 + b / m

L_op = LinearOperator((m * N, m * N), matvec=lambda v: v - (1 / m) * A_op @ v)


def get_L_and_psi():
    return L_op, psi
# -------------------------
# Output shapes for confirmation
# -------------------------
# print("L_op shape:", L_op.shape)
# print("psi shape:", psi.shape)

# plt.imshow(f.T, origin='lower', extent=[0, xmax, -vmax, vmax], aspect='auto')
# plt.title('Initial f(x, v, 0)')
# plt.xlabel('x')
# plt.ylabel('v')
# plt.colorbar()
# plt.show()


