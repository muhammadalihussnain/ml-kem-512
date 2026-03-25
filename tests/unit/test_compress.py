"""
Unit tests for Compression & Decompression (Milestone 3.3).

Tests cover:
- compress(d, x): output in [0, 2^d - 1]
- decompress(d, y): output in [0, Q-1]
- Decompress(Compress(x)) ≈ x with bounded error
- Error bound: |x - Decompress(Compress(x))| <= Q / 2^(d+1)
- Boundary values: 0, Q-1, Q//2
- ML-KEM-512 specific: d_u=10, d_v=4
- Polynomial-level wrappers: compress_poly, decompress_poly
- Roundtrip on full polynomials
"""

from ml_kem_512.encoding.compress import (
    DU,
    DV,
    compress,
    compress_poly,
    decompress,
    decompress_poly,
)
from ml_kem_512.polynomial.poly import N, Polynomial, Q

# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------


class TestParameters:
    def test_du_value(self):
        """DU = 10 for ML-KEM-512 ciphertext vector u."""
        assert DU == 10

    def test_dv_value(self):
        """DV = 4 for ML-KEM-512 ciphertext scalar v."""
        assert DV == 4


# ---------------------------------------------------------------------------
# compress() tests
# ---------------------------------------------------------------------------


class TestCompress:
    def test_output_in_range_du(self):
        """compress(DU, x) output is in [0, 2^DU - 1]."""
        for x in range(0, Q, 100):
            c = compress(DU, x)
            assert 0 <= c < (1 << DU), f"compress({DU}, {x}) = {c} out of range"

    def test_output_in_range_dv(self):
        """compress(DV, x) output is in [0, 2^DV - 1]."""
        for x in range(0, Q, 100):
            c = compress(DV, x)
            assert 0 <= c < (1 << DV), f"compress({DV}, {x}) = {c} out of range"

    def test_output_in_range_d1(self):
        """compress(1, x) output is 0 or 1."""
        for x in range(0, Q, 50):
            c = compress(1, x)
            assert c in (0, 1)

    def test_compress_zero(self):
        """compress(d, 0) = 0 for any d."""
        for d in [1, 4, 10, 12]:
            assert compress(d, 0) == 0

    def test_compress_reduces_size(self):
        """Compressed value fits in d bits (< 2^d)."""
        for d in [4, 10]:
            for x in range(0, Q, 33):
                assert compress(d, x) < (1 << d)

    def test_compress_boundary_q_minus_1(self):
        """compress(d, Q-1) is in valid range."""
        for d in [4, 10]:
            c = compress(d, Q - 1)
            assert 0 <= c < (1 << d)

    def test_compress_q_half(self):
        """compress(1, Q//2) = 1 (Q//2 is the midpoint, rounds to 1)."""
        assert compress(1, Q // 2) == 1

    def test_compress_all_coefficients_du(self):
        """All Q values compress correctly for d=DU."""
        for x in range(Q):
            c = compress(DU, x)
            assert 0 <= c < (1 << DU)


# ---------------------------------------------------------------------------
# decompress() tests
# ---------------------------------------------------------------------------


class TestDecompress:
    def test_output_in_range_du(self):
        """decompress(DU, y) output is in [0, Q-1]."""
        for y in range(1 << DU):
            v = decompress(DU, y)
            assert 0 <= v < Q, f"decompress({DU}, {y}) = {v} out of range"

    def test_output_in_range_dv(self):
        """decompress(DV, y) output is in [0, Q-1]."""
        for y in range(1 << DV):
            v = decompress(DV, y)
            assert 0 <= v < Q

    def test_decompress_zero(self):
        """decompress(d, 0) = 0 for any d."""
        for d in [1, 4, 10, 12]:
            assert decompress(d, 0) == 0

    def test_decompress_scales_up(self):
        """decompress(d, 2^d - 1) is close to Q-1."""
        for d in [4, 10]:
            v = decompress(d, (1 << d) - 1)
            assert v < Q
            # Should be close to Q (within one step)
            assert v > Q - Q // (1 << d) - 2


# ---------------------------------------------------------------------------
# Roundtrip error bound tests
# ---------------------------------------------------------------------------


class TestRoundtripErrorBound:
    def _signed_error(self, x: int, d: int) -> int:
        """Compute signed error: x - Decompress(Compress(x)), centered mod Q."""
        y = compress(d, x)
        x_approx = decompress(d, y)
        err = (x - x_approx) % Q
        # Center the error around 0
        if err > Q // 2:
            err -= Q
        return err

    def test_error_bounded_du(self):
        """
        For d=DU=10: |error| <= Q / 2^(DU+1) = 3329 / 2048 ~ 1.6
        So error must be at most 1 (integer bound).
        """
        bound = Q // (1 << (DU + 1)) + 1
        for x in range(Q):
            err = abs(self._signed_error(x, DU))
            assert err <= bound, f"x={x}: error={err} exceeds bound={bound}"

    def test_error_bounded_dv(self):
        """
        For d=DV=4: |error| <= Q / 2^(DV+1) = 3329 / 32 ~ 104.
        """
        bound = Q // (1 << (DV + 1)) + 1
        for x in range(Q):
            err = abs(self._signed_error(x, DV))
            assert err <= bound, f"x={x}: error={err} exceeds bound={bound}"

    def test_error_bounded_d1(self):
        """For d=1: |error| <= Q/4 ~ 832."""
        bound = Q // 4 + 1
        for x in range(Q):
            err = abs(self._signed_error(x, 1))
            assert err <= bound

    def test_zero_roundtrip_exact(self):
        """Compress(0) = 0, Decompress(0) = 0 — exact for all d."""
        for d in [1, 4, 10, 12]:
            assert decompress(d, compress(d, 0)) == 0

    def test_boundary_values(self):
        """Boundary values 0, Q-1, Q//2 roundtrip within error bound."""
        for d in [DV, DU]:
            bound = Q // (1 << (d + 1)) + 1
            for x in [0, Q - 1, Q // 2, 1, Q - 2]:
                err = abs(self._signed_error(x, d))
                assert err <= bound, f"d={d}, x={x}: error={err}"


# ---------------------------------------------------------------------------
# Polynomial-level tests
# ---------------------------------------------------------------------------


class TestCompressPoly:
    def test_compress_poly_output_length(self):
        """compress_poly returns a list of N=256 values."""
        poly = Polynomial(list(range(N)))
        result = compress_poly(DU, poly)
        assert len(result) == N

    def test_compress_poly_all_in_range(self):
        """All compressed values are in [0, 2^d - 1]."""
        poly = Polynomial([x % Q for x in range(N)])
        for d in [DV, DU]:
            result = compress_poly(d, poly)
            assert all(0 <= v < (1 << d) for v in result)

    def test_decompress_poly_output_is_polynomial(self):
        """decompress_poly returns a Polynomial."""
        values = [i % (1 << DU) for i in range(N)]
        result = decompress_poly(DU, values)
        assert isinstance(result, Polynomial)
        assert len(result.coeffs) == N

    def test_decompress_poly_all_in_range(self):
        """All decompressed coefficients are in [0, Q-1]."""
        for d in [DV, DU]:
            values = [i % (1 << d) for i in range(N)]
            result = decompress_poly(d, values)
            assert all(0 <= c < Q for c in result.coeffs)

    def test_poly_roundtrip_du(self):
        """Decompress(Compress(poly, DU)) ≈ poly with bounded error."""
        poly = Polynomial([x % Q for x in range(N)])
        compressed = compress_poly(DU, poly)
        recovered = decompress_poly(DU, compressed)
        bound = Q // (1 << (DU + 1)) + 1
        for orig, rec in zip(poly.coeffs, recovered.coeffs):
            err = (orig - rec) % Q
            if err > Q // 2:
                err -= Q
            assert abs(err) <= bound

    def test_poly_roundtrip_dv(self):
        """Decompress(Compress(poly, DV)) ≈ poly with bounded error."""
        poly = Polynomial([x % Q for x in range(N)])
        compressed = compress_poly(DV, poly)
        recovered = decompress_poly(DV, compressed)
        bound = Q // (1 << (DV + 1)) + 1
        for orig, rec in zip(poly.coeffs, recovered.coeffs):
            err = (orig - rec) % Q
            if err > Q // 2:
                err -= Q
            assert abs(err) <= bound

    def test_zero_poly_roundtrip(self):
        """Zero polynomial compresses and decompresses to zero."""
        from ml_kem_512.polynomial.poly import zero_poly

        z = zero_poly()
        for d in [DV, DU]:
            compressed = compress_poly(d, z)
            recovered = decompress_poly(d, compressed)
            assert recovered == z
