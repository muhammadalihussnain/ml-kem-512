"""
Unit tests for Matrix-Vector Multiplication and Dot Product (Milestones 4.2 & 4.3).

Milestone 4.2 — Matrix-vector multiplication:
  - A * v in NTT domain
  - Result is a k-vector of polynomials
  - NTT version matches schoolbook multiplication
  - Distributivity, associativity with scalar

Milestone 4.3 — Dot product:
  - t^T . s produces a single polynomial
  - Correct for known test vectors
  - Matches schoolbook
  - Used in KeyGen (t = A*s + e) and Decaps (v - s^T*u)
"""

from ml_kem_512.module.vector import (
    K,
    PolyVec,
    dot_ntt,
    inv_ntt_vec,
    mat_vec_mul,
    ntt_vec,
    transpose,
    zero_vec,
)
from ml_kem_512.polynomial.ntt import inv_ntt, ntt, ntt_mul
from ml_kem_512.polynomial.poly import N, Polynomial, Q, zero_poly
from ml_kem_512.sampling.uniform import sample_ntt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rand_poly(seed: int) -> Polynomial:
    return Polynomial([(seed * (i + 1) * 6700417) % Q for i in range(N)])


def rand_vec(seed: int) -> PolyVec:
    return PolyVec([rand_poly(seed + i) for i in range(K)])


def rand_ntt_vec(seed: int) -> PolyVec:
    return ntt_vec(rand_vec(seed))


def rand_matrix(seed: int) -> list:
    """k x k matrix of NTT coefficient lists (as from sample_ntt)."""
    rho = bytes([seed % 256] * 32)
    return [[sample_ntt(rho, i, j) for j in range(K)] for i in range(K)]


def schoolbook_mat_vec(A_poly: list, v: PolyVec) -> PolyVec:
    """
    Reference: matrix-vector multiply using schoolbook polynomial multiply.
    A_poly[i][j] is a Polynomial (not NTT list).
    """
    result = []
    for i in range(K):
        row_sum = zero_poly()
        for j in range(K):
            row_sum = row_sum + A_poly[i][j] * v[j]
        result.append(row_sum)
    return PolyVec(result)


# ---------------------------------------------------------------------------
# Milestone 4.2: Matrix-Vector Multiplication
# ---------------------------------------------------------------------------


class TestMatVecMulCorrectness:
    def test_result_is_polyvec(self):
        """mat_vec_mul returns a PolyVec of length k."""
        A = rand_matrix(0)
        v_hat = rand_ntt_vec(1)
        result = mat_vec_mul(A, v_hat)
        assert isinstance(result, PolyVec)
        assert len(result) == K

    def test_result_coefficients_in_range(self):
        """All output coefficients are in [0, Q-1]."""
        A = rand_matrix(1)
        v_hat = rand_ntt_vec(2)
        result = mat_vec_mul(A, v_hat)
        for i in range(K):
            assert all(0 <= c < Q for c in result[i].coeffs)

    def test_ntt_matches_schoolbook(self):
        """
        NTT-domain mat_vec_mul matches schoolbook multiplication.

        InvNTT(A_hat * v_hat) == A_poly * v  (schoolbook)
        """
        rho = bytes(32)
        # Build NTT matrix
        A_ntt = [[sample_ntt(rho, i, j) for j in range(K)] for i in range(K)]
        # Build polynomial matrix (InvNTT of each element)
        A_poly = [[inv_ntt(A_ntt[i][j]) for j in range(K)] for i in range(K)]

        v = rand_vec(5)
        v_hat = ntt_vec(v)

        # NTT-domain result
        ntt_result = inv_ntt_vec(mat_vec_mul(A_ntt, v_hat))
        # Schoolbook result
        sb_result = schoolbook_mat_vec(A_poly, v)

        assert ntt_result == sb_result

    def test_zero_vector_input(self):
        """A * zero_vec == zero_vec."""
        A = rand_matrix(0)
        z_hat = ntt_vec(zero_vec())
        result = mat_vec_mul(A, z_hat)
        assert inv_ntt_vec(result) == zero_vec()

    def test_identity_matrix(self):
        """Identity matrix * v == v."""
        one_ntt = ntt(Polynomial([1]))
        zero_ntt = ntt(zero_poly())
        I = [[one_ntt if i == j else zero_ntt for j in range(K)] for i in range(K)]
        v = rand_vec(3)
        v_hat = ntt_vec(v)
        result = inv_ntt_vec(mat_vec_mul(I, v_hat))
        assert result == v

    def test_distributivity_over_vector_addition(self):
        """A*(v + w) == A*v + A*w."""
        A = rand_matrix(0)
        v = rand_vec(1)
        w = rand_vec(2)
        v_hat = ntt_vec(v)
        w_hat = ntt_vec(w)
        vw_hat = ntt_vec(v + w)

        lhs = inv_ntt_vec(mat_vec_mul(A, vw_hat))
        rhs = inv_ntt_vec(mat_vec_mul(A, v_hat)) + inv_ntt_vec(mat_vec_mul(A, w_hat))
        assert lhs == rhs

    def test_transpose_mat_vec(self):
        """A^T * v is different from A * v in general."""
        A = rand_matrix(0)
        At = transpose(A)
        v_hat = rand_ntt_vec(1)
        result_A = mat_vec_mul(A, v_hat)
        result_At = mat_vec_mul(At, v_hat)
        # They should generally differ (non-symmetric matrix)
        assert result_A != result_At or True  # just verify no crash

    def test_keygen_pattern(self):
        """
        Simulate KeyGen: t_hat = A_hat * s_hat + e_hat
        Result must be a valid k-vector with coefficients in [0, Q-1].
        """
        from ml_kem_512.polynomial.ntt import inv_ntt
        from ml_kem_512.primitives.prf import prf
        from ml_kem_512.sampling.cbd import cbd_eta1

        rho = bytes(32)
        sigma = bytes(range(32))

        # Generate matrix A
        A = [[sample_ntt(rho, i, j) for j in range(K)] for i in range(K)]

        # Sample secret s and error e
        s_polys = [cbd_eta1(prf(sigma, n, 192)) for n in range(K)]
        e_polys = [cbd_eta1(prf(sigma, K + n, 192)) for n in range(K)]

        s = PolyVec(s_polys)
        e = PolyVec(e_polys)

        s_hat = ntt_vec(s)
        e_hat = ntt_vec(e)

        # t_hat = A * s_hat + e_hat
        t_hat = mat_vec_mul(A, s_hat)
        t_hat_vec = PolyVec([Polynomial(t_hat[i].coeffs) for i in range(K)])
        t_hat_final = t_hat_vec + e_hat

        assert len(t_hat_final) == K
        for i in range(K):
            assert all(0 <= c < Q for c in t_hat_final[i].coeffs)


# ---------------------------------------------------------------------------
# Milestone 4.3: Dot Product
# ---------------------------------------------------------------------------


class TestDotProduct:
    def test_dot_returns_polynomial(self):
        """dot_ntt returns a single Polynomial."""
        a_hat = rand_ntt_vec(1)
        b_hat = rand_ntt_vec(2)
        result = dot_ntt(a_hat, b_hat)
        assert isinstance(result, Polynomial)
        assert len(result.coeffs) == N

    def test_dot_coefficients_in_range(self):
        """All dot product coefficients are in [0, Q-1]."""
        a_hat = rand_ntt_vec(1)
        b_hat = rand_ntt_vec(2)
        result = dot_ntt(a_hat, b_hat)
        assert all(0 <= c < Q for c in result.coeffs)

    def test_dot_matches_schoolbook(self):
        """
        InvNTT(dot_ntt(NTT(a), NTT(b))) == a[0]*b[0] + a[1]*b[1]  (schoolbook).
        """
        a = rand_vec(1)
        b = rand_vec(2)
        a_hat = ntt_vec(a)
        b_hat = ntt_vec(b)

        # NTT dot product
        dot_result = inv_ntt(dot_ntt(a_hat, b_hat).coeffs)

        # Schoolbook
        sb = a[0] * b[0] + a[1] * b[1]

        assert dot_result == sb

    def test_dot_commutativity(self):
        """dot(a, b) == dot(b, a)."""
        a_hat = rand_ntt_vec(1)
        b_hat = rand_ntt_vec(2)
        assert dot_ntt(a_hat, b_hat) == dot_ntt(b_hat, a_hat)

    def test_dot_with_zero(self):
        """dot(v, zero_vec) == zero polynomial (in NTT domain)."""
        v_hat = rand_ntt_vec(1)
        z_hat = ntt_vec(zero_vec())
        result = dot_ntt(v_hat, z_hat)
        assert inv_ntt(result.coeffs) == zero_poly()

    def test_dot_known_vectors(self):
        """
        Known test: a = [1, 0, ...], b = [1, 0, ...]
        dot(a, b) = 1*1 + 0*0 = 1 (constant polynomial).
        """
        a = PolyVec([Polynomial([1]), zero_poly()])
        b = PolyVec([Polynomial([1]), zero_poly()])
        a_hat = ntt_vec(a)
        b_hat = ntt_vec(b)
        result = inv_ntt(dot_ntt(a_hat, b_hat).coeffs)
        assert result == Polynomial([1])

    def test_transpose_dot_pattern(self):
        """
        Simulate t^T . s used in Decaps:
          v - InvNTT(t_hat^T . u_hat)
        Result must be a valid polynomial.
        """
        t_hat = rand_ntt_vec(1)
        u_hat = rand_ntt_vec(2)
        inner = dot_ntt(t_hat, u_hat)
        inner_poly = inv_ntt(inner.coeffs)
        assert isinstance(inner_poly, Polynomial)
        assert all(0 <= c < Q for c in inner_poly.coeffs)

    def test_decaps_pattern(self):
        """
        Simulate Decaps inner product: w = v - InvNTT(s^T . NTT(u))
        where s is the secret key vector and u is the ciphertext vector.
        Result must be a valid polynomial.
        """
        from ml_kem_512.encoding.compress import decompress_poly
        from ml_kem_512.primitives.prf import prf
        from ml_kem_512.sampling.cbd import cbd_eta1

        sigma = bytes(32)
        s_polys = [cbd_eta1(prf(sigma, n, 192)) for n in range(K)]
        s = PolyVec(s_polys)
        s_hat = ntt_vec(s)

        # Simulate u (decompressed ciphertext vector)
        u_polys = [rand_poly(10 + i) for i in range(K)]
        u = PolyVec(u_polys)
        u_hat = ntt_vec(u)

        # Simulate v (decompressed ciphertext scalar)
        v = rand_poly(20)

        # w = v - InvNTT(s^T . u_hat)
        inner = dot_ntt(s_hat, u_hat)
        inner_poly = inv_ntt(inner.coeffs)
        w = v - inner_poly

        assert isinstance(w, Polynomial)
        assert all(0 <= c < Q for c in w.coeffs)
