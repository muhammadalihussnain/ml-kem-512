"""
Unit tests for Polynomial arithmetic (Milestone 2.1).

Tests cover:
- Construction: zero-padding, coefficient reduction, invalid sizes
- Addition in R_q (mod q, coefficient-wise)
- Subtraction in R_q (mod q, coefficient-wise)
- Negation
- Wrap-around at q boundary
- Algebraic properties: commutativity, associativity, identity, inverse
- Helpers: zero_poly, one_poly, is_zero, reduce
"""

import pytest

from ml_kem_512.polynomial.poly import N, Q, Polynomial, one_poly, zero_poly


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_is_zero(self):
        """Polynomial() creates the zero polynomial."""
        p = Polynomial()
        assert p.coeffs == [0] * N

    def test_length_is_always_n(self):
        """Every polynomial has exactly N=256 coefficients."""
        assert len(Polynomial()) == N
        assert len(Polynomial([1, 2, 3])) == N

    def test_short_input_zero_padded(self):
        """Coefficients shorter than N are zero-padded on the right."""
        p = Polynomial([1, 2, 3])
        assert p[0] == 1
        assert p[1] == 2
        assert p[2] == 3
        assert p[3] == 0
        assert p[255] == 0

    def test_full_length_input(self):
        """Exactly N coefficients are accepted."""
        coeffs = list(range(N))
        p = Polynomial(coeffs)
        assert p[0] == 0
        assert p[1] == 1

    def test_coefficients_reduced_mod_q(self):
        """Coefficients are automatically reduced into [0, Q-1]."""
        p = Polynomial([Q, Q + 1, -1, -Q])
        assert p[0] == 0          # Q mod Q = 0
        assert p[1] == 1          # (Q+1) mod Q = 1
        assert p[2] == Q - 1      # -1 mod Q = Q-1
        assert p[3] == 0          # -Q mod Q = 0

    def test_too_many_coefficients_raises(self):
        """More than N coefficients raises ValueError."""
        with pytest.raises(ValueError):
            Polynomial([0] * (N + 1))

    def test_zero_poly_helper(self):
        """zero_poly() returns the zero polynomial."""
        assert zero_poly() == Polynomial()

    def test_one_poly_helper(self):
        """one_poly() returns the constant polynomial 1."""
        p = one_poly()
        assert p[0] == 1
        assert all(p[i] == 0 for i in range(1, N))


# ---------------------------------------------------------------------------
# Addition tests
# ---------------------------------------------------------------------------


class TestAddition:
    def test_basic_addition(self):
        """Simple coefficient-wise addition."""
        p1 = Polynomial([1, 2, 3])
        p2 = Polynomial([4, 5, 6])
        result = p1 + p2
        assert result[0] == 5
        assert result[1] == 7
        assert result[2] == 9
        assert result[3] == 0

    def test_addition_wraps_mod_q(self):
        """Addition wraps around at q."""
        p1 = Polynomial([Q - 1])
        p2 = Polynomial([1])
        result = p1 + p2
        assert result[0] == 0  # (Q-1 + 1) mod Q = 0

    def test_addition_near_boundary(self):
        """Addition near q boundary stays in [0, Q-1]."""
        p1 = Polynomial([Q - 2])
        p2 = Polynomial([3])
        result = p1 + p2
        assert result[0] == 1  # (Q-2+3) mod Q = 1
        assert 0 <= result[0] < Q

    def test_add_zero_is_identity(self):
        """Adding zero polynomial leaves polynomial unchanged."""
        p = Polynomial(list(range(10)))
        assert p + zero_poly() == p

    def test_commutativity(self):
        """a + b == b + a."""
        a = Polynomial([1, 2, 3, 4])
        b = Polynomial([10, 20, 30, 40])
        assert a + b == b + a

    def test_associativity(self):
        """(a + b) + c == a + (b + c)."""
        a = Polynomial([1, 0, 2])
        b = Polynomial([0, 3, 0])
        c = Polynomial([4, 0, 5])
        assert (a + b) + c == a + (b + c)

    def test_all_coefficients_in_range_after_add(self):
        """All coefficients stay in [0, Q-1] after addition."""
        p1 = Polynomial([Q - 1] * N)
        p2 = Polynomial([Q - 1] * N)
        result = p1 + p2
        assert all(0 <= c < Q for c in result.coeffs)

    def test_add_known_example_from_plan(self):
        """
        From the implementation plan:
          p1 = [1, 2, 3, ..., 0]
          p2 = [4, 5, 6, ..., 0]
          p3 = p1 + p2 -> p3[0]=5, p3[1]=7, p3[2]=9
        """
        p1 = Polynomial([1, 2, 3])
        p2 = Polynomial([4, 5, 6])
        p3 = p1 + p2
        assert p3[0] == 5
        assert p3[1] == 7
        assert p3[2] == 9


# ---------------------------------------------------------------------------
# Subtraction tests
# ---------------------------------------------------------------------------


class TestSubtraction:
    def test_basic_subtraction(self):
        """Simple coefficient-wise subtraction."""
        p1 = Polynomial([5, 7, 9])
        p2 = Polynomial([4, 5, 6])
        result = p1 - p2
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3

    def test_subtraction_wraps_mod_q(self):
        """Subtraction wraps around correctly (no negative coefficients)."""
        p1 = Polynomial([0])
        p2 = Polynomial([1])
        result = p1 - p2
        assert result[0] == Q - 1  # 0 - 1 mod Q = Q-1

    def test_subtract_self_is_zero(self):
        """p - p == zero polynomial."""
        p = Polynomial([1, 2, 3, 100, 200])
        assert p - p == zero_poly()

    def test_subtract_zero_is_identity(self):
        """p - 0 == p."""
        p = Polynomial([10, 20, 30])
        assert p - zero_poly() == p

    def test_all_coefficients_in_range_after_sub(self):
        """All coefficients stay in [0, Q-1] after subtraction."""
        p1 = Polynomial([0] * N)
        p2 = Polynomial([Q - 1] * N)
        result = p1 - p2
        assert all(0 <= c < Q for c in result.coeffs)

    def test_subtraction_not_commutative(self):
        """a - b != b - a (unless a == b)."""
        a = Polynomial([1])
        b = Polynomial([2])
        assert a - b != b - a

    def test_add_sub_roundtrip(self):
        """(p + q) - q == p."""
        p = Polynomial([1, 2, 3, 4, 5])
        q = Polynomial([10, 20, 30, 40, 50])
        assert (p + q) - q == p


# ---------------------------------------------------------------------------
# Negation tests
# ---------------------------------------------------------------------------


class TestNegation:
    def test_neg_zero_is_zero(self):
        """-0 == 0."""
        assert -zero_poly() == zero_poly()

    def test_neg_one(self):
        """-1 coefficient becomes Q-1."""
        p = one_poly()
        neg_p = -p
        assert neg_p[0] == Q - 1
        assert all(neg_p[i] == 0 for i in range(1, N))

    def test_p_plus_neg_p_is_zero(self):
        """p + (-p) == 0."""
        p = Polynomial([1, 2, 3, 100, Q - 1])
        assert p + (-p) == zero_poly()

    def test_all_coefficients_in_range_after_neg(self):
        """All coefficients stay in [0, Q-1] after negation."""
        p = Polynomial(list(range(N)))
        neg_p = -p
        assert all(0 <= c < Q for c in neg_p.coeffs)


# ---------------------------------------------------------------------------
# Algebraic property tests
# ---------------------------------------------------------------------------


class TestAlgebraicProperties:
    def test_additive_identity(self):
        """p + 0 == p and 0 + p == p."""
        p = Polynomial([42, 100, 200])
        assert p + zero_poly() == p
        assert zero_poly() + p == p

    def test_additive_inverse(self):
        """p + (-p) == 0."""
        p = Polynomial([1, Q - 1, 500, 3000])
        assert p + (-p) == zero_poly()

    def test_double_negation(self):
        """--p == p."""
        p = Polynomial([1, 2, 3])
        assert -(-p) == p

    def test_subtraction_via_negation(self):
        """a - b == a + (-b)."""
        a = Polynomial([10, 20, 30])
        b = Polynomial([1, 2, 3])
        assert a - b == a + (-b)

    def test_coefficients_always_canonical(self):
        """After any operation, all coefficients are in [0, Q-1]."""
        p1 = Polynomial([Q - 1, 0, Q - 1])
        p2 = Polynomial([1, Q - 1, 1])
        for result in [p1 + p2, p1 - p2, -p1]:
            assert all(0 <= c < Q for c in result.coeffs)


# ---------------------------------------------------------------------------
# Equality and helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_equality(self):
        """Two polynomials with same coefficients are equal."""
        p1 = Polynomial([1, 2, 3])
        p2 = Polynomial([1, 2, 3])
        assert p1 == p2

    def test_inequality(self):
        """Polynomials with different coefficients are not equal."""
        assert Polynomial([1]) != Polynomial([2])

    def test_equality_with_non_polynomial_returns_not_implemented(self):
        """Comparing with a non-Polynomial returns NotImplemented."""
        p = Polynomial([1, 2, 3])
        assert p.__eq__(42) is NotImplemented
        assert p.__eq__("string") is NotImplemented

    def test_is_zero_true(self):
        """zero_poly().is_zero() is True."""
        assert zero_poly().is_zero()

    def test_is_zero_false(self):
        """Non-zero polynomial is_zero() is False."""
        assert not Polynomial([1]).is_zero()

    def test_reduce_returns_canonical(self):
        """reduce() returns polynomial with coefficients in [0, Q-1]."""
        p = Polynomial([Q + 5, -3])
        r = p.reduce()
        assert all(0 <= c < Q for c in r.coeffs)

    def test_getitem(self):
        """Indexing returns correct coefficient."""
        p = Polynomial([10, 20, 30])
        assert p[0] == 10
        assert p[1] == 20
        assert p[2] == 30

    def test_repr_does_not_crash(self):
        """repr() works for zero, small, and large non-zero polynomials."""
        repr(zero_poly())
        repr(Polynomial([1, 2, 3]))
        # trigger the '...' truncation branch (more than 8 non-zero terms)
        repr(Polynomial([1] * 20))

    def test_q_value(self):
        """Q is 3329 as per ML-KEM-512 spec."""
        assert Q == 3329

    def test_n_value(self):
        """N is 256 as per ML-KEM-512 spec."""
        assert N == 256
