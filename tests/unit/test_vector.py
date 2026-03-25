"""
Unit tests for Vector operations (Milestone 4.1).

Tests cover:
- PolyVec construction: k=2 polynomials, type checking
- Addition / subtraction / negation (component-wise)
- Algebraic properties: commutativity, associativity, identity, inverse
- ntt_vec / inv_ntt_vec: roundtrip InvNTT(NTT(v)) == v
- dot_ntt: dot product in NTT domain matches schoolbook
- mat_vec_mul: A*v in NTT domain
- transpose: A^T[i][j] == A[j][i]
- zero_vec helper
"""

import pytest

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
from ml_kem_512.polynomial.ntt import ntt, ntt_mul
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
    """k x k matrix of NTT coefficient lists."""
    rho = bytes([seed % 256] * 32)
    return [[sample_ntt(rho, i, j) for j in range(K)] for i in range(K)]


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestPolyVecConstruction:
    def test_k_value(self):
        """K = 2 for ML-KEM-512."""
        assert K == 2

    def test_create_from_two_polys(self):
        """PolyVec accepts exactly k=2 polynomials."""
        v = PolyVec([zero_poly(), zero_poly()])
        assert len(v) == K

    def test_wrong_length_raises(self):
        """PolyVec raises ValueError for wrong number of polynomials."""
        with pytest.raises(ValueError, match="k=2"):
            PolyVec([zero_poly()])

    def test_wrong_length_three_raises(self):
        """PolyVec raises ValueError for 3 polynomials."""
        with pytest.raises(ValueError, match="k=2"):
            PolyVec([zero_poly(), zero_poly(), zero_poly()])

    def test_wrong_type_raises(self):
        """PolyVec raises TypeError if element is not a Polynomial."""
        with pytest.raises(TypeError):
            PolyVec([zero_poly(), [1, 2, 3]])

    def test_getitem(self):
        """Indexing returns the correct polynomial."""
        p0 = rand_poly(1)
        p1 = rand_poly(2)
        v = PolyVec([p0, p1])
        assert v[0] == p0
        assert v[1] == p1

    def test_zero_vec(self):
        """zero_vec() returns a vector of zero polynomials."""
        v = zero_vec()
        assert v[0] == zero_poly()
        assert v[1] == zero_poly()

    def test_repr_does_not_crash(self):
        """repr() works without error."""
        repr(zero_vec())
        repr(rand_vec(1))

    def test_equality(self):
        """Two vectors with same polynomials are equal."""
        v1 = PolyVec([rand_poly(1), rand_poly(2)])
        v2 = PolyVec([rand_poly(1), rand_poly(2)])
        assert v1 == v2

    def test_inequality(self):
        """Vectors with different polynomials are not equal."""
        assert rand_vec(1) != rand_vec(2)

    def test_equality_non_polyvec(self):
        """Comparing with non-PolyVec returns NotImplemented."""
        v = zero_vec()
        assert v.__eq__(42) is NotImplemented


# ---------------------------------------------------------------------------
# Addition tests
# ---------------------------------------------------------------------------


class TestPolyVecAddition:
    def test_basic_addition(self):
        """Component-wise addition works correctly."""
        p0 = Polynomial([1, 2])
        p1 = Polynomial([3, 4])
        p2 = Polynomial([5, 6])
        p3 = Polynomial([7, 8])
        v1 = PolyVec([p0, p1])
        v2 = PolyVec([p2, p3])
        result = v1 + v2
        assert result[0] == p0 + p2
        assert result[1] == p1 + p3

    def test_add_zero_is_identity(self):
        """v + zero_vec == v."""
        v = rand_vec(5)
        assert v + zero_vec() == v

    def test_commutativity(self):
        """v + w == w + v."""
        v = rand_vec(1)
        w = rand_vec(2)
        assert v + w == w + v

    def test_associativity(self):
        """(v + w) + u == v + (w + u)."""
        v, w, u = rand_vec(1), rand_vec(2), rand_vec(3)
        assert (v + w) + u == v + (w + u)


# ---------------------------------------------------------------------------
# Subtraction tests
# ---------------------------------------------------------------------------


class TestPolyVecSubtraction:
    def test_basic_subtraction(self):
        """Component-wise subtraction works correctly."""
        v = rand_vec(1)
        w = rand_vec(2)
        result = v - w
        assert result[0] == v[0] - w[0]
        assert result[1] == v[1] - w[1]

    def test_subtract_self_is_zero(self):
        """v - v == zero_vec."""
        v = rand_vec(3)
        assert v - v == zero_vec()

    def test_subtract_zero_is_identity(self):
        """v - zero_vec == v."""
        v = rand_vec(4)
        assert v - zero_vec() == v

    def test_add_sub_roundtrip(self):
        """(v + w) - w == v."""
        v = rand_vec(1)
        w = rand_vec(2)
        assert (v + w) - w == v


# ---------------------------------------------------------------------------
# Negation tests
# ---------------------------------------------------------------------------


class TestPolyVecNegation:
    def test_neg_zero_is_zero(self):
        """-zero_vec == zero_vec."""
        assert -zero_vec() == zero_vec()

    def test_v_plus_neg_v_is_zero(self):
        """v + (-v) == zero_vec."""
        v = rand_vec(5)
        assert v + (-v) == zero_vec()

    def test_double_negation(self):
        """--v == v."""
        v = rand_vec(6)
        assert -(-v) == v


# ---------------------------------------------------------------------------
# NTT vector tests
# ---------------------------------------------------------------------------


class TestNTTVec:
    def test_ntt_vec_roundtrip(self):
        """InvNTT(NTT(v)) == v for each component."""
        v = rand_vec(1)
        assert inv_ntt_vec(ntt_vec(v)) == v

    def test_ntt_vec_zero(self):
        """NTT of zero vector stays zero after roundtrip."""
        assert inv_ntt_vec(ntt_vec(zero_vec())) == zero_vec()

    def test_ntt_vec_output_length(self):
        """NTT vector has K polynomials each with N coefficients."""
        v_hat = ntt_vec(rand_vec(1))
        assert len(v_hat) == K
        for i in range(K):
            assert len(v_hat[i].coeffs) == N


# ---------------------------------------------------------------------------
# Dot product tests
# ---------------------------------------------------------------------------


class TestDotNTT:
    def test_dot_zero_vec(self):
        """dot(v, zero_vec) == zero polynomial."""
        v_hat = rand_ntt_vec(1)
        z_hat = ntt_vec(zero_vec())
        result = dot_ntt(v_hat, z_hat)
        # InvNTT of zero NTT is zero poly
        from ml_kem_512.polynomial.ntt import inv_ntt

        assert inv_ntt(result.coeffs) == zero_poly()

    def test_dot_matches_schoolbook(self):
        """dot_ntt(NTT(a), NTT(b)) == NTT(a[0]*b[0] + a[1]*b[1])."""
        a = rand_vec(1)
        b = rand_vec(2)
        # NTT domain dot product
        a_hat = ntt_vec(a)
        b_hat = ntt_vec(b)
        dot_result = dot_ntt(a_hat, b_hat)

        # Schoolbook: a[0]*b[0] + a[1]*b[1]
        schoolbook = a[0] * b[0] + a[1] * b[1]
        # NTT of schoolbook result
        schoolbook_ntt = Polynomial(ntt(schoolbook))

        assert dot_result == schoolbook_ntt

    def test_dot_commutativity(self):
        """dot(a, b) == dot(b, a)."""
        a_hat = rand_ntt_vec(1)
        b_hat = rand_ntt_vec(2)
        assert dot_ntt(a_hat, b_hat) == dot_ntt(b_hat, a_hat)


# ---------------------------------------------------------------------------
# Matrix-vector multiplication tests
# ---------------------------------------------------------------------------


class TestMatVecMul:
    def test_mat_vec_output_is_polyvec(self):
        """mat_vec_mul returns a PolyVec."""
        A = rand_matrix(0)
        v_hat = rand_ntt_vec(1)
        result = mat_vec_mul(A, v_hat)
        assert isinstance(result, PolyVec)
        assert len(result) == K

    def test_mat_vec_zero_vector(self):
        """A * zero_vec == zero_vec (in NTT domain)."""
        A = rand_matrix(0)
        z_hat = ntt_vec(zero_vec())
        result = mat_vec_mul(A, z_hat)
        assert inv_ntt_vec(result) == zero_vec()

    def test_mat_vec_identity_matrix(self):
        """Identity matrix * v == v."""
        from ml_kem_512.polynomial.poly import one_poly

        # Identity: A[i][j] = NTT(1) if i==j else NTT(0)
        one_ntt = ntt(one_poly())
        zero_ntt = ntt(zero_poly())
        I = [[one_ntt if i == j else zero_ntt for j in range(K)] for i in range(K)]
        v = rand_vec(3)
        v_hat = ntt_vec(v)
        result = mat_vec_mul(I, v_hat)
        assert inv_ntt_vec(result) == v

    def test_mat_vec_output_coefficients_in_range(self):
        """All output coefficients are in [0, Q-1]."""
        A = rand_matrix(1)
        v_hat = rand_ntt_vec(2)
        result = mat_vec_mul(A, v_hat)
        for i in range(K):
            assert all(0 <= c < Q for c in result[i].coeffs)


# ---------------------------------------------------------------------------
# Transpose tests
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_transpose_swaps_indices(self):
        """A^T[i][j] == A[j][i]."""
        A = rand_matrix(0)
        At = transpose(A)
        for i in range(K):
            for j in range(K):
                assert At[i][j] == A[j][i]

    def test_double_transpose_is_identity(self):
        """(A^T)^T == A."""
        A = rand_matrix(0)
        assert transpose(transpose(A)) == A

    def test_transpose_size(self):
        """Transposed matrix has same k x k shape."""
        A = rand_matrix(0)
        At = transpose(A)
        assert len(At) == K
        assert all(len(row) == K for row in At)
