"""
Vector operations for ML-KEM-512.

A PolyVec is a vector of k polynomials in R_q.
For ML-KEM-512: k = 2.

Operations:
  - Addition:       v + w  (component-wise)
  - Subtraction:    v - w  (component-wise)
  - Negation:       -v
  - NTT forward:    ntt_vec(v)   -> PolyVec in NTT domain
  - NTT inverse:    inv_ntt_vec(v) -> PolyVec in poly domain
  - Dot product:    dot(v, w)    -> single Polynomial (NTT domain)
  - Matrix-vector:  mat_vec_mul(A, v) -> PolyVec (NTT domain)
  - Transpose mat:  transpose(A) -> transposed matrix
"""

from __future__ import annotations

from ml_kem_512.polynomial.ntt import inv_ntt, ntt, ntt_mul
from ml_kem_512.polynomial.poly import Polynomial, zero_poly

# ML-KEM-512 module rank
K = 2


class PolyVec:
    """
    Vector of k polynomials in R_q.

    For ML-KEM-512, k=2. Each element is a Polynomial.
    """

    def __init__(self, polys: list):
        """
        Create a polynomial vector.

        Args:
            polys: list of k Polynomial objects
        """
        if len(polys) != K:
            raise ValueError(f"PolyVec requires exactly k={K} polynomials, got {len(polys)}")
        for i, p in enumerate(polys):
            if not isinstance(p, Polynomial):
                raise TypeError(f"Element {i} must be a Polynomial, got {type(p)}")
        self.polys = list(polys)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: PolyVec) -> PolyVec:
        """Component-wise addition."""
        return PolyVec([self.polys[i] + other.polys[i] for i in range(K)])

    def __sub__(self, other: PolyVec) -> PolyVec:
        """Component-wise subtraction."""
        return PolyVec([self.polys[i] - other.polys[i] for i in range(K)])

    def __neg__(self) -> PolyVec:
        """Component-wise negation."""
        return PolyVec([-p for p in self.polys])

    # ------------------------------------------------------------------
    # Comparison / helpers
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolyVec):
            return NotImplemented
        return self.polys == other.polys

    def __repr__(self) -> str:
        return f"PolyVec([{', '.join(repr(p) for p in self.polys)}])"

    def __getitem__(self, index: int) -> Polynomial:
        return self.polys[index]

    def __len__(self) -> int:
        return K


def zero_vec() -> PolyVec:
    """Return the zero vector (all-zero polynomials)."""
    return PolyVec([zero_poly() for _ in range(K)])


def ntt_vec(v: PolyVec) -> PolyVec:
    """
    Apply forward NTT to each polynomial in the vector.

    Args:
        v: PolyVec in polynomial domain

    Returns:
        PolyVec in NTT domain (each element is a raw NTT list wrapped as Polynomial)
    """
    return PolyVec([Polynomial(ntt(v.polys[i])) for i in range(K)])


def inv_ntt_vec(v: PolyVec) -> PolyVec:
    """
    Apply inverse NTT to each polynomial in the vector.

    Args:
        v: PolyVec in NTT domain

    Returns:
        PolyVec in polynomial domain
    """
    return PolyVec([inv_ntt(v.polys[i].coeffs) for i in range(K)])


def dot_ntt(a: PolyVec, b: PolyVec) -> Polynomial:
    """
    Dot product of two NTT-domain vectors: sum_i ntt_mul(a[i], b[i]).

    Both vectors must already be in NTT domain.

    Args:
        a: PolyVec in NTT domain
        b: PolyVec in NTT domain

    Returns:
        Polynomial (NTT domain) = sum of pointwise products
    """
    result = [0] * 256
    for i in range(K):
        product = ntt_mul(a.polys[i].coeffs, b.polys[i].coeffs)
        result = [(result[j] + product[j]) % 3329 for j in range(256)]
    return Polynomial(result)


def mat_vec_mul(A: list, v: PolyVec) -> PolyVec:
    """
    Matrix-vector multiplication in NTT domain: A * v.

    A is a k x k matrix of NTT-domain polynomials (list of lists).
    v is a k-vector of NTT-domain polynomials.

    Result[i] = sum_j ntt_mul(A[i][j], v[j])

    Args:
        A: k x k matrix, A[i][j] is a list of 256 NTT coefficients
        v: PolyVec in NTT domain

    Returns:
        PolyVec in NTT domain
    """
    result = []
    for i in range(K):
        row_sum = [0] * 256
        for j in range(K):
            product = ntt_mul(A[i][j], v.polys[j].coeffs)
            row_sum = [(row_sum[k] + product[k]) % 3329 for k in range(256)]
        result.append(Polynomial(row_sum))
    return PolyVec(result)


def transpose(A: list) -> list:
    """
    Transpose a k x k matrix.

    Args:
        A: k x k matrix (list of lists of NTT coefficient lists)

    Returns:
        Transposed k x k matrix A^T where A^T[i][j] = A[j][i]
    """
    return [[A[j][i] for j in range(K)] for i in range(K)]
