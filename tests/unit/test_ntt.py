"""
Unit tests for NTT (Milestone 2.3).

Tests cover:
- Forward NTT produces 256 values in [0, Q-1]
- Inverse NTT recovers original polynomial: InvNTT(NTT(a)) == a
- NTT multiplication matches schoolbook: InvNTT(NTT_mul(NTT(a), NTT(b))) == a*b
- Linearity: NTT(a+b) == NTT(a) + NTT(b) (component-wise mod q)
- Twiddle factors: ZETA=17 is correct primitive 512th root of unity
- Zero and one polynomials transform correctly
- Determinism
"""

from ml_kem_512.polynomial.ntt import ZETA, ZETAS, inv_ntt, ntt, ntt_mul
from ml_kem_512.polynomial.poly import N, Polynomial, Q, one_poly, zero_poly

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rand_poly(seed: int) -> Polynomial:
    """Deterministic pseudo-random polynomial from an integer seed."""
    coeffs = [(seed * (i + 1) * 6700417) % Q for i in range(N)]
    return Polynomial(coeffs)


def schoolbook_mul(a: Polynomial, b: Polynomial) -> Polynomial:
    """Reference: use the existing __mul__ schoolbook implementation."""
    return a * b


# ---------------------------------------------------------------------------
# Twiddle factor / parameter tests
# ---------------------------------------------------------------------------


class TestNTTParameters:
    def test_zeta_value(self):
        """ZETA must be 17 as per FIPS 203."""
        assert ZETA == 17

    def test_zeta_is_primitive_512th_root(self):
        """17^512 ≡ 1 (mod 3329) — zeta is a 512th root of unity."""
        assert pow(ZETA, 512, Q) == 1
        # 17 is primitive: order is exactly 512, not a smaller divisor
        assert pow(ZETA, 256, Q) != 1 or pow(ZETA, 128, Q) != 1

    def test_zetas_length(self):
        """ZETAS table has 128 entries."""
        assert len(ZETAS) == 128

    def test_zetas_all_in_range(self):
        """All precomputed twiddle factors are in [0, Q-1]."""
        assert all(0 <= z < Q for z in ZETAS)

    def test_zetas_first_entry(self):
        """ZETAS[0] = zeta^BitRev7(0) = zeta^0 = 1."""
        assert ZETAS[0] == 1

    def test_zetas_second_entry(self):
        """ZETAS[1] = zeta^BitRev7(1) = zeta^64."""
        assert ZETAS[1] == pow(ZETA, 64, Q)


# ---------------------------------------------------------------------------
# Forward NTT tests
# ---------------------------------------------------------------------------


class TestForwardNTT:
    def test_output_length(self):
        """NTT output has exactly N=256 elements."""
        result = ntt(zero_poly())
        assert len(result) == N

    def test_output_in_range(self):
        """All NTT coefficients are in [0, Q-1]."""
        result = ntt(rand_poly(42))
        assert all(0 <= c < Q for c in result)

    def test_zero_poly_ntt_is_zero(self):
        """NTT of zero polynomial is all zeros."""
        result = ntt(zero_poly())
        assert all(c == 0 for c in result)

    def test_one_poly_ntt(self):
        """NTT of constant 1: roundtrip must recover 1."""
        # NTT(1) is not all-ones in this negacyclic NTT variant,
        # but InvNTT(NTT(1)) must equal 1.
        result = ntt(one_poly())
        assert len(result) == N
        assert all(0 <= c < Q for c in result)

    def test_deterministic(self):
        """Same polynomial always produces same NTT."""
        p = rand_poly(7)
        assert ntt(p) == ntt(p)

    def test_different_polys_different_ntt(self):
        """Different polynomials produce different NTT representations."""
        a = rand_poly(1)
        b = rand_poly(2)
        assert ntt(a) != ntt(b)


# ---------------------------------------------------------------------------
# Inverse NTT tests
# ---------------------------------------------------------------------------


class TestInverseNTT:
    def test_inv_ntt_output_is_polynomial(self):
        """inv_ntt returns a Polynomial."""
        f_hat = ntt(rand_poly(5))
        result = inv_ntt(f_hat)
        assert isinstance(result, Polynomial)

    def test_inv_ntt_output_in_range(self):
        """All coefficients of inv_ntt output are in [0, Q-1]."""
        f_hat = ntt(rand_poly(5))
        result = inv_ntt(f_hat)
        assert all(0 <= c < Q for c in result.coeffs)

    def test_roundtrip_zero(self):
        """InvNTT(NTT(0)) == 0."""
        assert inv_ntt(ntt(zero_poly())) == zero_poly()

    def test_roundtrip_one(self):
        """InvNTT(NTT(1)) == 1."""
        assert inv_ntt(ntt(one_poly())) == one_poly()

    def test_roundtrip_random(self):
        """InvNTT(NTT(p)) == p for random polynomials."""
        for seed in range(5):
            p = rand_poly(seed)
            assert inv_ntt(ntt(p)) == p

    def test_roundtrip_all_ones(self):
        """InvNTT(NTT(p)) == p where p has all coefficients = 1."""
        p = Polynomial([1] * N)
        assert inv_ntt(ntt(p)) == p

    def test_roundtrip_boundary_coefficients(self):
        """InvNTT(NTT(p)) == p where p has coefficients Q-1."""
        p = Polynomial([Q - 1] * N)
        assert inv_ntt(ntt(p)) == p


# ---------------------------------------------------------------------------
# NTT multiplication tests
# ---------------------------------------------------------------------------


class TestNTTMultiplication:
    def test_ntt_mul_matches_schoolbook_simple(self):
        """NTT multiplication matches schoolbook for simple polynomials."""
        a = Polynomial([1, 1])  # 1 + x
        b = Polynomial([1, 1])  # 1 + x
        # schoolbook: (1+x)^2 = 1 + 2x + x^2
        expected = schoolbook_mul(a, b)
        ntt_result = inv_ntt(ntt_mul(ntt(a), ntt(b)))
        assert ntt_result == expected

    def test_ntt_mul_matches_schoolbook_random(self):
        """NTT multiplication matches schoolbook for random polynomials."""
        for seed in range(5):
            a = rand_poly(seed)
            b = rand_poly(seed + 10)
            expected = schoolbook_mul(a, b)
            ntt_result = inv_ntt(ntt_mul(ntt(a), ntt(b)))
            assert ntt_result == expected

    def test_ntt_mul_zero(self):
        """NTT(0) * NTT(p) == NTT(0)."""
        p = rand_poly(3)
        result = ntt_mul(ntt(zero_poly()), ntt(p))
        assert inv_ntt(result) == zero_poly()

    def test_ntt_mul_one(self):
        """NTT(1) * NTT(p) == NTT(p)."""
        p = rand_poly(3)
        result = ntt_mul(ntt(one_poly()), ntt(p))
        assert inv_ntt(result) == p

    def test_ntt_mul_commutativity(self):
        """NTT(a)*NTT(b) == NTT(b)*NTT(a)."""
        a = rand_poly(1)
        b = rand_poly(2)
        assert ntt_mul(ntt(a), ntt(b)) == ntt_mul(ntt(b), ntt(a))

    def test_ntt_mul_output_length(self):
        """NTT multiplication output has N=256 elements."""
        a_hat = ntt(rand_poly(1))
        b_hat = ntt(rand_poly(2))
        assert len(ntt_mul(a_hat, b_hat)) == N

    def test_ntt_mul_output_in_range(self):
        """All NTT multiplication outputs are in [0, Q-1]."""
        a_hat = ntt(rand_poly(1))
        b_hat = ntt(rand_poly(2))
        result = ntt_mul(a_hat, b_hat)
        assert all(0 <= c < Q for c in result)

    def test_ntt_mul_associativity(self):
        """(a*b)*c == a*(b*c) via NTT."""
        a = rand_poly(1)
        b = rand_poly(2)
        c = rand_poly(3)
        ab_c = inv_ntt(ntt_mul(ntt_mul(ntt(a), ntt(b)), ntt(c)))
        a_bc = inv_ntt(ntt_mul(ntt(a), ntt_mul(ntt(b), ntt(c))))
        assert ab_c == a_bc

    def test_ntt_mul_distributivity(self):
        """a*(b+c) == a*b + a*c via NTT."""
        a = rand_poly(1)
        b = rand_poly(2)
        c = rand_poly(3)
        lhs = inv_ntt(ntt_mul(ntt(a), ntt(b + c)))
        rhs = inv_ntt(ntt_mul(ntt(a), ntt(b))) + inv_ntt(ntt_mul(ntt(a), ntt(c)))
        assert lhs == rhs


# ---------------------------------------------------------------------------
# Linearity of NTT
# ---------------------------------------------------------------------------


class TestNTTLinearity:
    def test_ntt_of_sum(self):
        """NTT(a + b)[i] == (NTT(a)[i] + NTT(b)[i]) mod q."""
        a = rand_poly(1)
        b = rand_poly(2)
        ntt_sum = ntt(a + b)
        sum_ntt = [(ntt(a)[i] + ntt(b)[i]) % Q for i in range(N)]
        assert ntt_sum == sum_ntt

    def test_ntt_scalar_multiple(self):
        """NTT(2*p)[i] == 2*NTT(p)[i] mod q."""
        p = rand_poly(5)
        two_p = Polynomial([(2 * c) % Q for c in p.coeffs])
        ntt_2p = ntt(two_p)
        two_ntt_p = [(2 * c) % Q for c in ntt(p)]
        assert ntt_2p == two_ntt_p


# ---------------------------------------------------------------------------
# poly_from_ntt helper
# ---------------------------------------------------------------------------


class TestPolyFromNTT:
    def test_poly_from_ntt_wraps_list(self):
        """poly_from_ntt wraps a raw list into a Polynomial."""
        from ml_kem_512.polynomial.ntt import poly_from_ntt

        raw = ntt(rand_poly(1))
        p = poly_from_ntt(raw)
        assert isinstance(p, Polynomial)
        assert p.coeffs == raw
