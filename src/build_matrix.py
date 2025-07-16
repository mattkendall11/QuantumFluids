import numpy as np
from vlasov import Nx,Nv, gamma, A_op, N
import numpy.linalg as LA
from scipy.sparse.linalg import LinearOperator, gmres



'''
Defining L requires defining N, M1 and M2
'''
def M1(k, A, h):
    """
    Return a LinearOperator representing
    M1 = sum_{j=0}^{k-1} |j+1><j| \otimes (A h)/(j+1)
    Compatible with both dense and LinearOperator A.
    """
    # Dimension of the first subsystem
    dim = k + 1
    # Get dimension of matrix A for second subsystem
    if hasattr(A, 'shape'):
        A_dim = A.shape[0]
    else:
        raise ValueError("A must have a shape attribute")
    total_dim = dim * A_dim

    def matvec(v):
        v = np.asarray(v).reshape((dim, A_dim))
        out = np.zeros((dim, A_dim), dtype=complex)
        for j in range(k):
            # Apply (A h)/(j+1) to the j-th block
            block_in = v[j]
            if isinstance(A, LinearOperator):
                block_out = A.matvec(block_in) * h / (j + 1)
            else:
                block_out = (A * h / (j + 1)) @ block_in
            out[j + 1] += block_out
        return out.reshape(-1)

    print(f"[M1] Creating LinearOperator for M1 with shape ({total_dim}, {total_dim})")
    return LinearOperator((total_dim, total_dim), matvec=matvec, dtype=complex)

def M2(k, A_dim=None):
    """
    Return a LinearOperator representing
    M2 = sum_{j=0}^k |0><j| \otimes I
    This maps each block j to the first block (|0>).
    Compatible with both dense and LinearOperator A.
    """
    dim = k + 1
    if A_dim is None:
        A_dim = N  # fallback, but should be passed explicitly for generality
    total_dim = dim * A_dim

    def matvec(v):
        v = np.asarray(v).reshape((dim, A_dim))
        out = np.zeros((dim, A_dim), dtype=complex)
        # M2 = sum_{j=0}^k |0><j| \otimes I
        # This means: for each j, take the j-th block and add it to the 0-th block
        for j in range(dim):
            out[0] += v[j]
        return out.reshape(-1)

    print(f"[M2] Creating LinearOperator for M2 with shape ({total_dim}, {total_dim})")
    return LinearOperator((total_dim, total_dim), matvec=matvec, dtype=complex)


def N_op(m, p, M1, M2):
    """
    Return a LinearOperator representing
    N = sum_{i=0}^m |i+1><i| \otimes M2 (I-M1)^{-1} + sum_{i=m+1}^{m+p-1} |i+1><i| \otimes I
    Compatible with LinearOperator M1, M2.
    """
    # Extract dimensions
    total_dim = M1.shape[0]
    n_dim = m + p + 1
    second_dim = total_dim // (m + 1)
    N_total_dim = n_dim * second_dim

    # Precompute LinearOperator for (I - M1)^{-1} using GMRES
    def apply_inv_M1(v):
        # Solve (I - M1)x = v for x
        x, info = gmres(LinearOperator(M1.shape, matvec=lambda x: x - M1.matvec(x), dtype=complex), v, atol=1e-10)
        if info != 0:
            print(f"[N_op] Warning: GMRES did not fully converge, info={info}")
        return x
    Inv_M1 = LinearOperator(M1.shape, matvec=apply_inv_M1, dtype=complex)

    def matvec(v):
        v = np.asarray(v).reshape((n_dim, second_dim))
        out = np.zeros((n_dim, second_dim), dtype=complex)
        # First sum: i=0 to m
        for i in range(m + 1):
            block_in = v[i]
            # Apply (I-M1)^{-1}
            inv_block = Inv_M1.matvec(block_in)
            # Apply M2
            m2_block = M2.matvec(inv_block)
            out[i + 1] += m2_block[:second_dim]  # Only add the first block (|0>), as M2 maps to first block
        # Second sum: i=m+1 to m+p-1
        for i in range(m + 1, m + p):
            out[i + 1] += v[i]
        return out.reshape(-1)

    print(f"[N_op] Creating LinearOperator for N_op with shape ({N_total_dim}, {N_total_dim})")
    return LinearOperator((N_total_dim, N_total_dim), matvec=matvec, dtype=complex)

def implement_l(N_op):
    """
    Return a LinearOperator representing L = I - N_op,
    compatible with N_op as a LinearOperator.
    """
    shape = N_op.shape
    def matvec(v):
        return v - N_op.matvec(v)
    print(f"[implement_l] Creating LinearOperator for L with shape {shape}")
    return LinearOperator(shape, matvec=matvec, dtype=complex)


print(gamma)
# For an 8x8 grid, m*N is small enough for dense matrix operations
print(A_op.shape)
# A_matrix = A_op @ np.eye(A_op.shape[1])
t1 = M1(5, A_op, 1)
t2 = M2(5)
t3 = N_op(5, 5, t1, t2)
t4 = implement_l(t3)

print(t4.shape)

# -------------------------
# Test functions for M1 and M2
# -------------------------
def test_M1_M2():
    """
    Test functions to verify M1 and M2 are working correctly.
    """
    print("\n=== Testing M1 and M2 ===")
    
    # Create a simple test matrix
    test_A = np.array([[1, 2], [3, 4]], dtype=complex)
    k = 2  # 3 blocks (0, 1, 2)
    h = 0.5
    
    print(f"Test matrix A:\n{test_A}")
    print(f"k={k}, h={h}")
    
    # Test M1
    print("\n--- Testing M1 ---")
    m1_op = M1(k, test_A, h)
    
    # Create test input: [v0, v1, v2] where each vj is a 2-vector
    test_input = np.array([1+0j, 2+0j,  # v0
                          3+0j, 4+0j,  # v1  
                          5+0j, 6+0j], dtype=complex)  # v2
    
    print(f"Test input: {test_input}")
    print(f"Reshaped as blocks: {test_input.reshape(k+1, -1)}")
    
    # Apply M1
    m1_output = m1_op.matvec(test_input)
    print(f"M1 output: {m1_output}")
    print(f"M1 output reshaped: {m1_output.reshape(k+1, -1)}")
    
    # Let's debug what M1 is actually doing
    print("\nDebugging M1:")
    v = test_input.reshape(k+1, -1)
    print(f"Input blocks: v[0]={v[0]}, v[1]={v[1]}, v[2]={v[2]}")
    
    # Manual calculation
    out = np.zeros((k+1, 2), dtype=complex)
    for j in range(k):
        block_in = v[j]
        block_out = (test_A * h / (j + 1)) @ block_in
        print(f"j={j}: (A * h/{j+1}) * v[{j}] = ({test_A * h / (j + 1)}) @ {block_in} = {block_out}")
        out[j + 1] += block_out
    print(f"Manual calculation result: {out.flatten()}")
    
    # Expected M1 behavior: M1 = sum_{j=0}^{k-1} |j+1><j| \otimes (A h)/(j+1)
    # So: out[1] = (A * h/1) * v0 = (A * 0.5) * [1,2] = [0.5, 1.5, 1.5, 2.5] * [1,2] = [2.5, 4.5]
    #     out[2] = (A * h/2) * v1 = (A * 0.25) * [3,4] = [0.25, 0.75, 0.75, 1.25] * [3,4] = [1.75, 3.25]
    expected_m1 = np.array([0+0j, 0+0j,      # out[0] = 0
                           2.5+0j, 4.5+0j,   # out[1] = (A*h/1)*v0
                           1.75+0j, 3.25+0j], dtype=complex)  # out[2] = (A*h/2)*v1
    
    print(f"Expected M1 output: {expected_m1}")
    print(f"M1 correct: {np.allclose(m1_output, expected_m1, atol=1e-10)}")
    
    # Test M2
    print("\n--- Testing M2 ---")
    m2_op = M2(k, A_dim=2)
    
    # Apply M2
    m2_output = m2_op.matvec(test_input)
    print(f"M2 output: {m2_output}")
    print(f"M2 output reshaped: {m2_output.reshape(k+1, -1)}")
    
    # Expected M2 behavior: M2 = sum_{j=0}^k |0><j| \otimes I
    # So: out[0] = v0 + v1 + v2 = [1,2] + [3,4] + [5,6] = [9,12]
    #     out[1] = 0, out[2] = 0
    expected_m2 = np.array([9+0j, 12+0j,     # out[0] = sum of all blocks
                           0+0j, 0+0j,       # out[1] = 0
                           0+0j, 0+0j], dtype=complex)  # out[2] = 0
    
    print(f"Expected M2 output: {expected_m2}")
    print(f"M2 correct: {np.allclose(m2_output, expected_m2, atol=1e-10)}")
    
    return m1_op, m2_op

# Run the tests
if __name__ == "__main__":
    test_M1_M2()

