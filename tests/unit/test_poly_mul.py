"""
Unit tests for Polynomial multiplication (Milestone 2.2).

Tests cover:
- Basic schoolbook multiplication
- Reduction modulo (x^256 + 1): x^256 = -1 mod q
- Algebraic properties: commutativity, associativity, distributivity
- Identity and zero element
- The critical plan test: x^256 = q-1
- All coefficients stay in [0, q-1] after multiplication
"""

import pytest

from ml_kem_512.polynomial.poly import N, Polynomial, Q, one_poly, zero_poly

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def x_poly(power: int = 1) -> Polynomial:
    """Return the monomial x^power."""
    if power >= N:
        raise ValueError("Use repeated multiplication for power >= N")
    coeffs = [0] * N
    coeffs[power] = 1
    return Polynomial(coeffs)


# ---------------------------------------------------------------------------
# Basic multiplication
# ---------------------------------------------------------------------------


class TestBasicMultiplication:
    def test_zero_times_anything_is_zero(self):
        """0 * p == 0."""
        p = Polynomial([1, 2, 3, 4])
        assert zero_poly() * p == zero_poly()
        assert p * zero_poly() == zero_poly()

    def test_one_times_anything_is_itself(self):
        """1 * p == p."""
        p = Polynomial([1, 2, 3, 4])
        assert one_poly() * p == p
        assert p * one_poly() == p

    def test_constant_times_constant(self):
        """2 * 3 = 6 (constant polynomials)."""
        two = Polynomial([2])
        three = Polynomial([3])
        result = two * three
        assert result[0] == 6
        assert all(result[i] == 0 for i in range(1, N))

    def test_constant_multiplication_mod_q(self):
        """Constant product wraps mod q."""
        # (Q-1) * 2 = 2Q-2 ≡ Q-2 (mod Q)
        a = Polynomial([Q - 1])
        b = Polynomial([2])
        result = a * b
        assert result[0] == (Q - 1) * 2 % Q

    def test_x_times_x_is_x_squared(self):
        """x * x = x^2."""
        x = x_poly(1)
        result = x * x
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 1
        assert all(result[i] == 0 for i in range(3, N))

    def test_x_times_x_squared_is_x_cubed(self):
        """x * x^2 = x^3."""
        x = x_poly(1)
        x2 = x_poly(2)
        result = x * x2
        assert result[3] == 1
        assert all(result[i] == 0 for i in [0, 1, 2] + list(range(4, N)))

    def test_multiply_two_linear_polynomials(self):
        """(1 + x)(1 + x) = 1 + 2x + x^2."""
        p = Polynomial([1, 1])  # 1 + x
        result = p * p
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 1
        assert all(result[i] == 0 for i in range(3, N))

    def test_multiply_known_example(self):
        """(2 + 3x)(4 + 5x) = 8 + 22x + 15x^2."""
        a = Polynomial([2, 3])
        b = Polynomial([4, 5])
        result = a * b
        assert result[0] == 8
        assert result[1] == 22
        assert result[2] == 15
        assert all(result[i] == 0 for i in range(3, N))


# ---------------------------------------------------------------------------
# Reduction modulo x^256 + 1
# ---------------------------------------------------------------------------


class TestReductionModXN:
    def test_x255_times_x_wraps(self):
        """x^255 * x = x^256 = -1 = q-1 (constant term)."""
        x255 = x_poly(255)
        x1 = x_poly(1)
        result = x255 * x1
        assert result[0] == Q - 1  # x^256 ≡ -1 ≡ Q-1
        assert all(result[i] == 0 for i in range(1, N))

    def test_x255_times_x2_wraps(self):
        """x^255 * x^2 = x^257 = x^256 * x = -x."""
        x255 = x_poly(255)
        x2 = x_poly(2)
        result = x255 * x2
        # x^257 = x^256 * x = -1 * x = -x => coeff[1] = Q-1
        assert result[0] == 0
        assert result[1] == Q - 1
        assert all(result[i] == 0 for i in range(2, N))

    def test_critical_x_power_256(self):
        """
        Critical test from the plan:
          x = [0, 1, 0, ..., 0]
          x^256 (by repeated multiplication) = [Q-1, 0, ..., 0]
        """
        x = x_poly(1)
        result = one_poly()
        for _ in range(N):  # multiply x by itself 256 times
            result = result * x
        # x^256 ≡ -1 ≡ Q-1 (mod Q) in the constant term
        assert result[0] == Q - 1
        assert all(result[i] == 0 for i in range(1, N))

    def test_x128_squared_is_neg_one(self):
        """(x^128)^2 = x^256 = -1."""
        x128 = x_poly(128)
        result = x128 * x128
        assert result[0] == Q - 1
        assert all(result[i] == 0 for i in range(1, N))

    def test_wraparound_coefficients_in_range(self):
        """After wraparound multiplication, all coefficients in [0, Q-1]."""
        x255 = x_poly(255)
        result = x255 * x255  # x^510 = x^256 * x^254 = -x^254
        assert all(0 <= c < Q for c in result.coeffs)


# ---------------------------------------------------------------------------
# Algebraic properties
# ---------------------------------------------------------------------------


class TestMultiplicationProperties:
    def test_commutativity(self):
        """a * b == b * a."""
        a = Polynomial([1, 2, 3, 4])
        b = Polynomial([5, 6, 7, 8])
        assert a * b == b * a

    def test_commutativity_with_wraparound(self):
        """Commutativity holds even when reduction wraps coefficients."""
        a = Polynomial([0] * 200 + [1])  # x^200
        b = Polynomial([0] * 100 + [1])  # x^100
        assert a * b == b * a

    def test_associativity(self):
        """(a * b) * c == a * (b * c)."""
        a = Polynomial([1, 1])
        b = Polynomial([1, 2])
        c = Polynomial([3, 1])
        assert (a * b) * c == a * (b * c)

    def test_distributivity_left(self):
        """a * (b + c) == a*b + a*c."""
        a = Polynomial([1, 2])
        b = Polynomial([3, 4])
        c = Polynomial([5, 6])
        assert a * (b + c) == a * b + a * c

    def test_distributivity_right(self):
        """(a + b) * c == a*c + b*c."""
        a = Polynomial([1, 2])
        b = Polynomial([3, 4])
        c = Polynomial([5, 6])
        assert (a + b) * c == a * c + b * c

    def test_all_coefficients_in_range_after_mul(self):
        """All coefficients stay in [0, Q-1] after multiplication."""
        a = Polynomial([Q - 1] * N)
        b = Polynomial([Q - 1] * N)
        result = a * b
        assert all(0 <= c < Q for c in result.coeffs)

    def test_multiply_then_add(self):
        """(a * b) + (a * c) == a * (b + c) (distributivity via addition)."""
        a = Polynomial(list(range(10)))
        b = Polynomial(list(range(10, 20)))
        c = Polynomial(list(range(20, 30)))
        assert a * b + a * c == a * (b + c)

    def test_square_is_commutative_with_self(self):
        """p * p is the same regardless of order."""
        p = Polynomial([1, 2, 3, 4, 5])
        assert p * p == p * p

    def test_negation_and_multiplication(self):
        """(-a) * b == -(a * b) == a * (-b)."""
        a = Polynomial([1, 2, 3])
        b = Polynomial([4, 5, 6])
        assert (-a) * b == -(a * b)
        assert a * (-b) == -(a * b)
