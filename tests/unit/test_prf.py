"""
Unit tests for PRF and XOF (Milestone 1.2).

Tests cover:
- PRF(s, b): SHAKE-256 based pseudorandom function
- XOF(ρ, i, j): SHAKE-128 based extendable output function

Verification strategy:
- Determinism: same inputs → same outputs
- Domain separation: different (i,j) pairs → different outputs
- Cross-check: PRF output matches raw SHAKE-256(s || b)
- Cross-check: XOF output matches raw SHAKE-128(ρ || i || j)
- Input validation: wrong sizes raise errors
- Output length: correct byte counts
"""

import hashlib

import pytest

from ml_kem_512.primitives.prf import XOF, prf, xof

# ---------------------------------------------------------------------------
# Helpers — ground-truth reference implementations (no abstraction)
# ---------------------------------------------------------------------------


def _ref_prf(s: bytes, b: int, length: int) -> bytes:
    """Raw SHAKE-256(s || b) — used to cross-verify prf()."""
    h = hashlib.shake_256()
    h.update(s)
    h.update(bytes([b]))
    return h.digest(length)


def _ref_xof(rho: bytes, i: int, j: int, length: int) -> bytes:
    """Raw SHAKE-128(ρ || i || j) — used to cross-verify xof()."""
    h = hashlib.shake_128()
    h.update(rho)
    h.update(bytes([i]))
    h.update(bytes([j]))
    return h.digest(length)


# ---------------------------------------------------------------------------
# PRF Tests
# ---------------------------------------------------------------------------


class TestPRF:
    """Tests for PRF(s, b) — SHAKE-256 based pseudorandom function."""

    SEED = bytes(range(32))  # 0x00..0x1f — fixed seed for reproducibility

    def test_output_length_64(self):
        """PRF with η=1 needs 64 bytes (64*1)."""
        result = prf(self.SEED, 0, 64)
        assert len(result) == 64

    def test_output_length_128(self):
        """PRF with η=2 needs 128 bytes (64*2)."""
        result = prf(self.SEED, 0, 128)
        assert len(result) == 128

    def test_output_length_192(self):
        """PRF with η=3 needs 192 bytes (64*3)."""
        result = prf(self.SEED, 0, 192)
        assert len(result) == 192

    def test_deterministic(self):
        """Same (s, b) always produces same output."""
        r1 = prf(self.SEED, 5, 64)
        r2 = prf(self.SEED, 5, 64)
        assert r1 == r2

    def test_different_counter_different_output(self):
        """Different b values produce different outputs (domain separation)."""
        outputs = [prf(self.SEED, b, 64) for b in range(8)]
        # All outputs must be unique
        assert len(set(outputs)) == 8

    def test_different_seed_different_output(self):
        """Different seeds produce different outputs."""
        seed1 = bytes(range(32))
        seed2 = bytes(range(1, 33))
        assert prf(seed1, 0, 64) != prf(seed2, 0, 64)

    def test_cross_check_with_raw_shake256(self):
        """PRF output must exactly match SHAKE-256(s || b)."""
        for b in range(5):
            assert prf(self.SEED, b, 128) == _ref_prf(self.SEED, b, 128)

    def test_counter_boundary_values(self):
        """PRF works at counter boundaries 0 and 255."""
        r0 = prf(self.SEED, 0, 64)
        r255 = prf(self.SEED, 255, 64)
        assert len(r0) == 64
        assert len(r255) == 64
        assert r0 != r255

    def test_invalid_seed_length(self):
        """PRF must reject seeds that are not 32 bytes."""
        with pytest.raises(ValueError, match="32 bytes"):
            prf(b"short", 0, 64)

    def test_invalid_counter_negative(self):
        """PRF must reject negative counter."""
        with pytest.raises(ValueError):
            prf(self.SEED, -1, 64)

    def test_invalid_counter_too_large(self):
        """PRF must reject counter > 255."""
        with pytest.raises(ValueError):
            prf(self.SEED, 256, 64)

    def test_keygen_usage_pattern(self):
        """
        Simulate ML-KEM KeyGen usage:
        s[0] = CBD(PRF(σ, 0)), s[1] = CBD(PRF(σ, 1))
        e[0] = CBD(PRF(σ, 2)), e[1] = CBD(PRF(σ, 3))
        All 4 outputs must be distinct.
        """
        sigma = bytes(32)  # all-zero seed
        outputs = [prf(sigma, n, 192) for n in range(4)]
        assert len(set(outputs)) == 4


# ---------------------------------------------------------------------------
# XOF Tests
# ---------------------------------------------------------------------------


class TestXOF:
    """Tests for XOF(ρ, i, j) — SHAKE-128 based extendable output function."""

    RHO = bytes(range(32))  # fixed 32-byte seed

    def test_output_length(self):
        """XOF produces exactly the requested number of bytes."""
        for length in [32, 64, 168, 504]:
            result = xof(self.RHO, 0, 0, length)
            assert len(result) == length

    def test_deterministic(self):
        """Same (ρ, i, j) always produces same output."""
        r1 = xof(self.RHO, 1, 0, 168)
        r2 = xof(self.RHO, 1, 0, 168)
        assert r1 == r2

    def test_domain_separation_by_i(self):
        """Different row index i → different output."""
        r0 = xof(self.RHO, 0, 0, 168)
        r1 = xof(self.RHO, 1, 0, 168)
        assert r0 != r1

    def test_domain_separation_by_j(self):
        """Different column index j → different output."""
        r0 = xof(self.RHO, 0, 0, 168)
        r1 = xof(self.RHO, 0, 1, 168)
        assert r0 != r1

    def test_domain_separation_all_matrix_positions(self):
        """All 4 matrix positions (i,j) for k=2 produce distinct outputs."""
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        outputs = [xof(self.RHO, i, j, 168) for i, j in positions]
        assert len(set(outputs)) == 4

    def test_different_rho_different_output(self):
        """Different ρ seeds produce different outputs."""
        rho1 = bytes(range(32))
        rho2 = bytes(range(1, 33))
        assert xof(rho1, 0, 0, 168) != xof(rho2, 0, 0, 168)

    def test_cross_check_with_raw_shake128(self):
        """XOF output must exactly match SHAKE-128(ρ || i || j)."""
        for i in range(2):
            for j in range(2):
                assert xof(self.RHO, i, j, 504) == _ref_xof(self.RHO, i, j, 504)

    def test_xof_class_matches_xof_function(self):
        """XOF class and xof() convenience function produce identical output."""
        stream = XOF(self.RHO, 1, 1)
        assert stream.read(168) == xof(self.RHO, 1, 1, 168)

    def test_invalid_rho_length(self):
        """XOF must reject ρ that is not 32 bytes."""
        with pytest.raises(ValueError, match="32 bytes"):
            XOF(b"tooshort", 0, 0)

    def test_invalid_index_i(self):
        """XOF must reject i outside [0, 255]."""
        with pytest.raises(ValueError):
            XOF(self.RHO, 256, 0)

    def test_invalid_index_j(self):
        """XOF must reject j outside [0, 255]."""
        with pytest.raises(ValueError):
            XOF(self.RHO, 0, -1)

    def test_matrix_generation_pattern(self):
        """
        Simulate ML-KEM matrix generation:
        A[i][j] = SampleNTT(XOF(ρ, i, j))
        Verify all 4 streams for k=2 are independent.
        """
        k = 2
        streams = {}
        for i in range(k):
            for j in range(k):
                streams[(i, j)] = xof(self.RHO, i, j, 504)

        # All streams must be unique
        values = list(streams.values())
        assert len(set(values)) == k * k

    def test_xof_output_is_not_prf_output(self):
        """XOF (SHAKE-128) and PRF (SHAKE-256) must produce different outputs."""
        seed = bytes(range(32))
        xof_out = xof(seed, 0, 0, 64)
        prf_out = prf(seed, 0, 64)
        assert xof_out != prf_out

    def test_seed_returns_rho(self):
        """XOF.seed() returns the original rho used to initialize the stream."""
        stream = XOF(self.RHO, 0, 0)
        assert stream.seed() == self.RHO
