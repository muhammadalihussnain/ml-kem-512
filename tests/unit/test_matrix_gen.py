"""
Unit tests for Matrix Generation (Milestone 5.1).

Tests cover:
- Matrix shape: k x k = 2 x 2 for ML-KEM-512
- Each element: list of N=256 NTT coefficients in [0, Q-1]
- Determinism: same rho -> same matrix
- Domain separation: different rho -> different matrix
- A[i][j] != A[j][i] in general (non-symmetric)
- Transpose flag: generate_matrix(rho, transpose=True)[i][j] == A[j][i]
- A and A^T are consistent: A^T[i][j] == A[j][i]
- Input validation: wrong rho length raises ValueError
- Integration: matrix can be used directly in mat_vec_mul
"""

import pytest

from ml_kem_512.module.vector import K, PolyVec, inv_ntt_vec, mat_vec_mul, ntt_vec
from ml_kem_512.pke.matrix import generate_matrix
from ml_kem_512.polynomial.poly import N, Polynomial, Q, zero_poly

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RHO = bytes(range(32))
RHO2 = bytes(range(1, 33))


# ---------------------------------------------------------------------------
# Shape and type tests
# ---------------------------------------------------------------------------


class TestMatrixShape:
    def test_matrix_is_k_by_k(self):
        """Matrix has k rows and k columns."""
        A = generate_matrix(RHO)
        assert len(A) == K
        assert all(len(row) == K for row in A)

    def test_each_element_is_list(self):
        """Each matrix element is a list."""
        A = generate_matrix(RHO)
        for i in range(K):
            for j in range(K):
                assert isinstance(A[i][j], list)

    def test_each_element_length_n(self):
        """Each matrix element has exactly N=256 coefficients."""
        A = generate_matrix(RHO)
        for i in range(K):
            for j in range(K):
                assert len(A[i][j]) == N

    def test_all_coefficients_in_range(self):
        """All NTT coefficients are in [0, Q-1]."""
        A = generate_matrix(RHO)
        for i in range(K):
            for j in range(K):
                assert all(0 <= c < Q for c in A[i][j])

    def test_no_coefficient_equals_q(self):
        """No coefficient equals Q (rejection sampling worked)."""
        A = generate_matrix(RHO)
        for i in range(K):
            for j in range(K):
                assert Q not in A[i][j]


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestMatrixDeterminism:
    def test_same_rho_same_matrix(self):
        """Same rho always produces the same matrix."""
        A1 = generate_matrix(RHO)
        A2 = generate_matrix(RHO)
        for i in range(K):
            for j in range(K):
                assert A1[i][j] == A2[i][j]

    def test_different_rho_different_matrix(self):
        """Different rho produces different matrix."""
        A1 = generate_matrix(RHO)
        A2 = generate_matrix(RHO2)
        # At least one element must differ
        any_diff = any(A1[i][j] != A2[i][j] for i in range(K) for j in range(K))
        assert any_diff

    def test_all_four_elements_deterministic(self):
        """All 4 elements are individually deterministic."""
        A = generate_matrix(RHO)
        A_again = generate_matrix(RHO)
        for i in range(K):
            for j in range(K):
                assert A[i][j] == A_again[i][j]


# ---------------------------------------------------------------------------
# Domain separation tests
# ---------------------------------------------------------------------------


class TestMatrixDomainSeparation:
    def test_elements_are_distinct(self):
        """All 4 matrix elements A[i][j] are distinct."""
        A = generate_matrix(RHO)
        elements = [tuple(A[i][j]) for i in range(K) for j in range(K)]
        assert len(set(elements)) == K * K

    def test_not_symmetric(self):
        """A[0][1] != A[1][0] in general (non-symmetric matrix)."""
        A = generate_matrix(RHO)
        assert A[0][1] != A[1][0]

    def test_diagonal_elements_differ(self):
        """A[0][0] != A[1][1]."""
        A = generate_matrix(RHO)
        assert A[0][0] != A[1][1]


# ---------------------------------------------------------------------------
# Transpose tests
# ---------------------------------------------------------------------------


class TestMatrixTranspose:
    def test_transpose_flag_swaps_indices(self):
        """generate_matrix(rho, transpose=True)[i][j] == A[j][i]."""
        A = generate_matrix(RHO)
        At = generate_matrix(RHO, transpose=True)
        for i in range(K):
            for j in range(K):
                assert At[i][j] == A[j][i]

    def test_transpose_shape(self):
        """Transposed matrix has same k x k shape."""
        At = generate_matrix(RHO, transpose=True)
        assert len(At) == K
        assert all(len(row) == K for row in At)

    def test_transpose_coefficients_in_range(self):
        """All transposed matrix coefficients are in [0, Q-1]."""
        At = generate_matrix(RHO, transpose=True)
        for i in range(K):
            for j in range(K):
                assert all(0 <= c < Q for c in At[i][j])

    def test_double_transpose_is_original(self):
        """(A^T)^T == A."""
        A = generate_matrix(RHO)
        At = generate_matrix(RHO, transpose=True)
        # At[i][j] = A[j][i], so At^T[i][j] = At[j][i] = A[i][j]
        for i in range(K):
            for j in range(K):
                assert At[j][i] == A[i][j]


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestMatrixValidation:
    def test_wrong_rho_length_raises(self):
        """generate_matrix raises ValueError for rho != 32 bytes."""
        with pytest.raises(ValueError, match="32 bytes"):
            generate_matrix(bytes(16))

    def test_empty_rho_raises(self):
        """generate_matrix raises ValueError for empty rho."""
        with pytest.raises(ValueError, match="32 bytes"):
            generate_matrix(b"")

    def test_33_byte_rho_raises(self):
        """generate_matrix raises ValueError for 33-byte rho."""
        with pytest.raises(ValueError, match="32 bytes"):
            generate_matrix(bytes(33))


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestMatrixIntegration:
    def test_matrix_usable_in_mat_vec_mul(self):
        """Matrix from generate_matrix works directly in mat_vec_mul."""
        from ml_kem_512.polynomial.poly import zero_poly

        A = generate_matrix(RHO)
        v = PolyVec([Polynomial([1]), zero_poly()])
        v_hat = ntt_vec(v)
        result = mat_vec_mul(A, v_hat)
        assert isinstance(result, PolyVec)
        assert len(result) == K
        for i in range(K):
            assert all(0 <= c < Q for c in result[i].coeffs)

    def test_keygen_matrix_generation(self):
        """
        Simulate KeyGen Step 1-2:
          (rho, sigma) = G(d)
          A = generate_matrix(rho)
        Matrix must be 2x2 with valid NTT coefficients.
        """
        from ml_kem_512.primitives.kdf import G

        d = bytes(range(32))
        rho, sigma = G(d + bytes([2]))  # k=2 appended
        A = generate_matrix(rho)

        assert len(A) == K
        for i in range(K):
            for j in range(K):
                assert len(A[i][j]) == N
                assert all(0 <= c < Q for c in A[i][j])

    def test_encaps_uses_transpose(self):
        """
        Encaps uses A^T: generate_matrix(rho, transpose=True).
        A^T must be valid and differ from A.
        """
        A = generate_matrix(RHO)
        At = generate_matrix(RHO, transpose=True)

        # A and A^T are different (non-symmetric)
        assert A[0][1] != At[0][1]

        # Both are valid
        for i in range(K):
            for j in range(K):
                assert len(At[i][j]) == N
                assert all(0 <= c < Q for c in At[i][j])
