"""
Unit tests for Centered Binomial Distribution sampling (Milestone 3.1).

Tests cover:
- Output: 256 coefficients, all in [0, Q-1]
- Coefficient range: values represent [-eta, eta] reduced mod Q
- Determinism: same bytes -> same polynomial
- Input validation: wrong byte length raises ValueError
- Statistical: mean ~ 0, variance ~ eta/2 over many samples
- Both eta=2 (ETA2) and eta=3 (ETA1) variants
- Integration with PRF: cbd(prf(sigma, N)) works end-to-end
"""

import math

import pytest

from ml_kem_512.polynomial.poly import N, Q
from ml_kem_512.primitives.prf import prf
from ml_kem_512.sampling.cbd import ETA1, ETA2, _bytes_to_bits, cbd, cbd_eta1, cbd_eta2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIGMA = bytes(range(32))  # fixed 32-byte seed


def _canonical_to_signed(c: int, eta: int) -> int:
    """Convert a coefficient in [0, Q-1] back to signed [-eta, eta]."""
    if c > Q // 2:
        return c - Q
    return c


# ---------------------------------------------------------------------------
# Basic output tests
# ---------------------------------------------------------------------------


class TestCBDOutput:
    def test_output_is_polynomial_eta2(self):
        """cbd_eta2 returns a Polynomial with N=256 coefficients."""
        data = bytes(128)
        result = cbd_eta2(data)
        assert len(result.coeffs) == N

    def test_output_is_polynomial_eta1(self):
        """cbd_eta1 returns a Polynomial with N=256 coefficients."""
        data = bytes(192)
        result = cbd_eta1(data)
        assert len(result.coeffs) == N

    def test_all_coefficients_in_range_eta2(self):
        """All coefficients are in [0, Q-1] for eta=2."""
        data = bytes(range(128))
        result = cbd_eta2(data)
        assert all(0 <= c < Q for c in result.coeffs)

    def test_all_coefficients_in_range_eta1(self):
        """All coefficients are in [0, Q-1] for eta=3."""
        data = bytes(i % 256 for i in range(192))
        result = cbd_eta1(data)
        assert all(0 <= c < Q for c in result.coeffs)

    def test_signed_range_eta2(self):
        """Signed coefficients are in [-eta2, eta2] = [-2, 2]."""
        data = bytes(range(128))
        result = cbd_eta2(data)
        for c in result.coeffs:
            signed = _canonical_to_signed(c, ETA2)
            assert -ETA2 <= signed <= ETA2

    def test_signed_range_eta1(self):
        """Signed coefficients are in [-eta1, eta1] = [-3, 3]."""
        data = bytes(i % 256 for i in range(192))
        result = cbd_eta1(data)
        for c in result.coeffs:
            signed = _canonical_to_signed(c, ETA1)
            assert -ETA1 <= signed <= ETA1


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestCBDDeterminism:
    def test_deterministic_eta2(self):
        """Same bytes always produce same polynomial for eta=2."""
        data = bytes(range(128))
        assert cbd_eta2(data) == cbd_eta2(data)

    def test_deterministic_eta1(self):
        """Same bytes always produce same polynomial for eta=3."""
        data = bytes(i % 256 for i in range(192))
        assert cbd_eta1(data) == cbd_eta1(data)

    def test_different_bytes_different_poly(self):
        """Different input bytes produce different polynomials."""
        data1 = bytes(128)
        data2 = bytes([1] * 128)
        assert cbd_eta2(data1) != cbd_eta2(data2)

    def test_cbd_generic_matches_eta2(self):
        """cbd(2, data) == cbd_eta2(data)."""
        data = bytes(range(128))
        assert cbd(2, data) == cbd_eta2(data)

    def test_cbd_generic_matches_eta1(self):
        """cbd(3, data) == cbd_eta1(data)."""
        data = bytes(i % 256 for i in range(192))
        assert cbd(3, data) == cbd_eta1(data)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestCBDValidation:
    def test_wrong_length_eta2_raises(self):
        """Wrong byte length for eta=2 raises ValueError."""
        with pytest.raises(ValueError, match="128"):
            cbd_eta2(bytes(64))

    def test_wrong_length_eta1_raises(self):
        """Wrong byte length for eta=3 raises ValueError."""
        with pytest.raises(ValueError, match="192"):
            cbd_eta1(bytes(64))

    def test_invalid_eta_raises(self):
        """eta values other than 2 or 3 raise ValueError."""
        with pytest.raises(ValueError, match="eta=2 or eta=3"):
            cbd(1, bytes(64))

    def test_invalid_eta_4_raises(self):
        """eta=4 raises ValueError."""
        with pytest.raises(ValueError, match="eta=2 or eta=3"):
            cbd(4, bytes(256))

    def test_empty_bytes_raises(self):
        """Empty bytes raises ValueError."""
        with pytest.raises(ValueError):
            cbd_eta2(b"")


# ---------------------------------------------------------------------------
# Known-value test (all-zero input)
# ---------------------------------------------------------------------------


class TestCBDKnownValues:
    def test_all_zero_bytes_eta2(self):
        """All-zero input: all bits are 0, so a=0, b=0, coeff=0 for all."""
        result = cbd_eta2(bytes(128))
        assert all(c == 0 for c in result.coeffs)

    def test_all_zero_bytes_eta1(self):
        """All-zero input: all coefficients are 0."""
        result = cbd_eta1(bytes(192))
        assert all(c == 0 for c in result.coeffs)

    def test_all_ones_bytes_eta2(self):
        """All-0xFF input: all bits are 1, so a=eta, b=eta, coeff=0 for all."""
        result = cbd_eta2(bytes([0xFF] * 128))
        assert all(c == 0 for c in result.coeffs)

    def test_all_ones_bytes_eta1(self):
        """All-0xFF input: a=eta, b=eta, coeff=0 for all."""
        result = cbd_eta1(bytes([0xFF] * 192))
        assert all(c == 0 for c in result.coeffs)

    def test_alternating_bits_eta2(self):
        """
        0x55 = 0b01010101: alternating 1,0,1,0...
        For eta=2, each coeff uses 4 bits: bits [1,0,1,0] -> a=1+0=1, b=1+0=1 -> coeff=0
        """
        result = cbd_eta2(bytes([0x55] * 128))
        assert all(c == 0 for c in result.coeffs)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


class TestCBDStatistics:
    def _collect_signed_samples(self, eta: int, n_polys: int) -> list:
        """Generate n_polys polynomials and collect all signed coefficients."""
        samples = []
        byte_len = 64 * eta
        for i in range(n_polys):
            seed = bytes([i % 256] * 32)
            data = prf(seed, i % 256, byte_len)
            poly = cbd(eta, data)
            for c in poly.coeffs:
                samples.append(_canonical_to_signed(c, eta))
        return samples

    def test_mean_near_zero_eta2(self):
        """Mean of CBD_2 samples is approximately 0 (within 0.1)."""
        samples = self._collect_signed_samples(ETA2, 40)
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, f"Mean too far from 0: {mean}"

    def test_mean_near_zero_eta1(self):
        """Mean of CBD_3 samples is approximately 0 (within 0.1)."""
        samples = self._collect_signed_samples(ETA1, 40)
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.1, f"Mean too far from 0: {mean}"

    def test_variance_eta2(self):
        """Variance of CBD_2 ~ eta/2 = 1.0 (within 20%)."""
        samples = self._collect_signed_samples(ETA2, 40)
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)
        expected_var = ETA2 / 2  # = 1.0
        assert (
            abs(variance - expected_var) < 0.2 * expected_var
        ), f"Variance {variance:.3f} too far from expected {expected_var}"

    def test_variance_eta1(self):
        """Variance of CBD_3 ~ eta/2 = 1.5 (within 20%)."""
        samples = self._collect_signed_samples(ETA1, 40)
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)
        expected_var = ETA1 / 2  # = 1.5
        assert (
            abs(variance - expected_var) < 0.2 * expected_var
        ), f"Variance {variance:.3f} too far from expected {expected_var}"

    def test_all_values_in_range_eta2(self):
        """All sampled values are in [-2, 2] for eta=2."""
        samples = self._collect_signed_samples(ETA2, 10)
        assert all(-ETA2 <= s <= ETA2 for s in samples)

    def test_all_values_in_range_eta1(self):
        """All sampled values are in [-3, 3] for eta=3."""
        samples = self._collect_signed_samples(ETA1, 10)
        assert all(-ETA1 <= s <= ETA1 for s in samples)


# ---------------------------------------------------------------------------
# PRF integration test
# ---------------------------------------------------------------------------


class TestCBDWithPRF:
    def test_keygen_pattern_eta1(self):
        """
        Simulate KeyGen: s[i] = CBD_eta1(PRF(sigma, i)) for i=0,1
        Both polynomials must have coefficients in [-3, 3].
        """
        sigma = bytes(32)
        for counter in range(2):
            data = prf(sigma, counter, 64 * ETA1)
            poly = cbd_eta1(data)
            assert len(poly.coeffs) == N
            for c in poly.coeffs:
                assert -ETA1 <= _canonical_to_signed(c, ETA1) <= ETA1

    def test_encaps_pattern_eta2(self):
        """
        Simulate Encaps: r[i] = CBD_eta2(PRF(r_seed, i)) for i=0,1
        Both polynomials must have coefficients in [-2, 2].
        """
        r_seed = bytes(range(32))
        for counter in range(2):
            data = prf(r_seed, counter, 64 * ETA2)
            poly = cbd_eta2(data)
            assert len(poly.coeffs) == N
            for c in poly.coeffs:
                assert -ETA2 <= _canonical_to_signed(c, ETA2) <= ETA2

    def test_prf_cbd_deterministic(self):
        """PRF -> CBD pipeline is fully deterministic."""
        sigma = bytes(32)
        data = prf(sigma, 0, 64 * ETA1)
        p1 = cbd_eta1(data)
        p2 = cbd_eta1(data)
        assert p1 == p2


# ---------------------------------------------------------------------------
# Bit conversion helper test
# ---------------------------------------------------------------------------


class TestBytesToBits:
    def test_zero_byte(self):
        """0x00 -> 8 zero bits."""
        bits = _bytes_to_bits(bytes([0x00]))
        assert bits == [0] * 8

    def test_ff_byte(self):
        """0xFF -> 8 one bits."""
        bits = _bytes_to_bits(bytes([0xFF]))
        assert bits == [1] * 8

    def test_lsb_first(self):
        """0x01 -> [1, 0, 0, 0, 0, 0, 0, 0] (LSB first)."""
        bits = _bytes_to_bits(bytes([0x01]))
        assert bits[0] == 1
        assert all(b == 0 for b in bits[1:])

    def test_length(self):
        """Output length is 8 * input length."""
        for n in [1, 16, 32, 128]:
            assert len(_bytes_to_bits(bytes(n))) == 8 * n
