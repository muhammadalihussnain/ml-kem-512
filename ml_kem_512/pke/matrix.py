"""
Matrix generation for ML-KEM-512 K-PKE.

Implements FIPS 203 KeyGen Step 2:

  For i in 0..k-1:
    For j in 0..k-1:
      A[i][j] = SampleNTT(rho, i, j)

Note: FIPS 203 Section 5.1 uses XOF(rho, i, j) for A[i][j].
The matrix is generated in NTT domain and used directly in
matrix-vector multiplication without further transformation.

Each A[i][j] is a list of 256 NTT coefficients in [0, Q-1].
"""

from __future__ import annotations

from ml_kem_512.module.vector import K
from ml_kem_512.polynomial.poly import Q
from ml_kem_512.sampling.uniform import sample_ntt


def generate_matrix(rho: bytes, transpose: bool = False) -> list:
    """
    Generate the public matrix A (or A^T) from seed rho.

    FIPS 203 Section 5.1:
      A[i][j] = SampleNTT(rho, i, j)

    For encryption (Encaps), A^T is needed:
      A^T[i][j] = A[j][i] = SampleNTT(rho, j, i)

    Args:
        rho: 32-byte public seed (from G(d))
        transpose: if True, generate A^T instead of A

    Returns:
        k x k matrix where each element is a list of 256 NTT coefficients
    """
    if len(rho) != 32:
        raise ValueError(f"rho must be 32 bytes, got {len(rho)}")

    A = []
    for i in range(K):
        row = []
        for j in range(K):
            if transpose:
                # A^T[i][j] = A[j][i] = SampleNTT(rho, j, i)
                row.append(sample_ntt(rho, j, i))
            else:
                # A[i][j] = SampleNTT(rho, i, j)
                row.append(sample_ntt(rho, i, j))
        A.append(row)
    return A
