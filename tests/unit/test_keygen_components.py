"""
Unit tests for Secret Vector Generation and KeyGen components (Milestone 5.2).

Tests cover:
- sample_secret_vector: k polynomials from CBD_eta1(PRF(sigma, i))
- sample_error_vector:  k polynomials from CBD_eta1(PRF(sigma, k+i))
- Coefficients in range [-eta1, eta1] = [-3, 3]
- NTT transformation of secret/error vectors
- compute_public_key_ntt: t_hat = A*s_hat + e_hat
- encode/decode public key: 800-byte roundtrip
- encode/decode secret key: 768-byte roundtrip
- Counter separation: s and e use different PRF counters
- Determinism from sigma seed
"""

import pytest

from ml_kem_512.module.vector import K, PolyVec, inv_ntt_vec, ntt_vec
from ml_kem_512.pke.keygen import (
    compute_public_key_ntt,
    decode_public_key,
    decode_secret_key,
    encode_public_key,
    encode_secret_key,
    sample_error_vector,
    sample_secret_vector,
)
from ml_kem_512.pke.matrix import generate_matrix
from ml_kem_512.polynomial.poly import N, Polynomial, Q
from ml_kem_512.sampling.cbd import ETA1

SIGMA = bytes(range(32))
RHO = bytes(range(32))


def _signed(c: int) -> int:
    """Convert canonical [0,Q-1] coefficient to signed."""
    return c - Q if c > Q // 2 else c


# ---------------------------------------------------------------------------
# sample_secret_vector tests
# ---------------------------------------------------------------------------


class TestSampleSecretVector:
    def test_returns_polyvec(self):
        """sample_secret_vector returns a PolyVec."""
        s = sample_secret_vector(SIGMA)
        assert isinstance(s, PolyVec)

    def test_length_is_k(self):
        """Secret vector has exactly k=2 polynomials."""
        s = sample_secret_vector(SIGMA)
        assert len(s) == K

    def test_each_poly_has_n_coefficients(self):
        """Each polynomial has N=256 coefficients."""
        s = sample_secret_vector(SIGMA)
        for i in range(K):
            assert len(s[i].coeffs) == N

    def test_coefficients_in_canonical_range(self):
        """All coefficients are in [0, Q-1]."""
        s = sample_secret_vector(SIGMA)
        for i in range(K):
            assert all(0 <= c < Q for c in s[i].coeffs)

    def test_signed_coefficients_in_eta1_range(self):
        """Signed coefficients are in [-eta1, eta1] = [-3, 3]."""
        s = sample_secret_vector(SIGMA)
        for i in range(K):
            for c in s[i].coeffs:
                assert -ETA1 <= _signed(c) <= ETA1

    def test_deterministic(self):
        """Same sigma always produces same secret vector."""
        s1 = sample_secret_vector(SIGMA)
        s2 = sample_secret_vector(SIGMA)
        assert s1 == s2

    def test_different_sigma_different_vector(self):
        """Different sigma produces different secret vector."""
        s1 = sample_secret_vector(SIGMA)
        s2 = sample_secret_vector(bytes(range(1, 33)))
        assert s1 != s2

    def test_invalid_sigma_raises(self):
        """Wrong sigma length raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            sample_secret_vector(bytes(16))

    def test_ntt_roundtrip(self):
        """InvNTT(NTT(s)) == s."""
        s = sample_secret_vector(SIGMA)
        s_hat = ntt_vec(s)
        assert inv_ntt_vec(s_hat) == s

    def test_counter_start_zero(self):
        """Default counter_start=0 uses PRF counters 0 and 1."""
        s = sample_secret_vector(SIGMA)
        # Verify by manually computing with counter 0 and 1
        from ml_kem_512.primitives.prf import prf
        from ml_kem_512.sampling.cbd import cbd_eta1

        p0 = cbd_eta1(prf(SIGMA, 0, 64 * ETA1))
        p1 = cbd_eta1(prf(SIGMA, 1, 64 * ETA1))
        assert s[0] == p0
        assert s[1] == p1


# ---------------------------------------------------------------------------
# sample_error_vector tests
# ---------------------------------------------------------------------------


class TestSampleErrorVector:
    def test_returns_polyvec(self):
        """sample_error_vector returns a PolyVec."""
        e = sample_error_vector(SIGMA)
        assert isinstance(e, PolyVec)

    def test_length_is_k(self):
        """Error vector has exactly k=2 polynomials."""
        assert len(sample_error_vector(SIGMA)) == K

    def test_signed_coefficients_in_eta1_range(self):
        """Signed coefficients are in [-eta1, eta1]."""
        e = sample_error_vector(SIGMA)
        for i in range(K):
            for c in e[i].coeffs:
                assert -ETA1 <= _signed(c) <= ETA1

    def test_deterministic(self):
        """Same sigma always produces same error vector."""
        assert sample_error_vector(SIGMA) == sample_error_vector(SIGMA)

    def test_error_differs_from_secret(self):
        """Error vector uses different PRF counters than secret vector."""
        s = sample_secret_vector(SIGMA)
        e = sample_error_vector(SIGMA)
        assert s != e

    def test_counter_offset_k(self):
        """Error vector uses PRF counters k and k+1 (i.e. 2 and 3)."""
        e = sample_error_vector(SIGMA)
        from ml_kem_512.primitives.prf import prf
        from ml_kem_512.sampling.cbd import cbd_eta1

        p0 = cbd_eta1(prf(SIGMA, K + 0, 64 * ETA1))
        p1 = cbd_eta1(prf(SIGMA, K + 1, 64 * ETA1))
        assert e[0] == p0
        assert e[1] == p1


# ---------------------------------------------------------------------------
# compute_public_key_ntt tests
# ---------------------------------------------------------------------------


class TestComputePublicKeyNTT:
    def _setup(self):
        A = generate_matrix(RHO)
        s = sample_secret_vector(SIGMA)
        e = sample_error_vector(SIGMA)
        s_hat = ntt_vec(s)
        e_hat = ntt_vec(e)
        return A, s_hat, e_hat

    def test_returns_polyvec(self):
        """compute_public_key_ntt returns a PolyVec."""
        A, s_hat, e_hat = self._setup()
        t_hat = compute_public_key_ntt(A, s_hat, e_hat)
        assert isinstance(t_hat, PolyVec)

    def test_length_is_k(self):
        """t_hat has k=2 polynomials."""
        A, s_hat, e_hat = self._setup()
        t_hat = compute_public_key_ntt(A, s_hat, e_hat)
        assert len(t_hat) == K

    def test_coefficients_in_range(self):
        """All t_hat coefficients are in [0, Q-1]."""
        A, s_hat, e_hat = self._setup()
        t_hat = compute_public_key_ntt(A, s_hat, e_hat)
        for i in range(K):
            assert all(0 <= c < Q for c in t_hat[i].coeffs)

    def test_deterministic(self):
        """Same inputs always produce same t_hat."""
        A, s_hat, e_hat = self._setup()
        t1 = compute_public_key_ntt(A, s_hat, e_hat)
        t2 = compute_public_key_ntt(A, s_hat, e_hat)
        assert t1 == t2

    def test_error_changes_result(self):
        """Different error vector produces different t_hat."""
        A, s_hat, e_hat = self._setup()
        _, _, e_hat2 = (
            generate_matrix(RHO),
            ntt_vec(sample_secret_vector(bytes(range(1, 33)))),
            ntt_vec(sample_error_vector(bytes(range(1, 33)))),
        )
        t1 = compute_public_key_ntt(A, s_hat, e_hat)
        t2 = compute_public_key_ntt(A, s_hat, e_hat2)
        assert t1 != t2


# ---------------------------------------------------------------------------
# encode/decode public key tests
# ---------------------------------------------------------------------------


class TestPublicKeyEncoding:
    def _make_t_hat(self):
        A = generate_matrix(RHO)
        s_hat = ntt_vec(sample_secret_vector(SIGMA))
        e_hat = ntt_vec(sample_error_vector(SIGMA))
        return compute_public_key_ntt(A, s_hat, e_hat)

    def test_encoded_length_800(self):
        """Encoded public key is exactly 800 bytes."""
        t_hat = self._make_t_hat()
        ek = encode_public_key(t_hat, RHO)
        assert len(ek) == 800

    def test_rho_appended_at_end(self):
        """Last 32 bytes of encoded public key is rho."""
        t_hat = self._make_t_hat()
        ek = encode_public_key(t_hat, RHO)
        assert ek[768:] == RHO

    def test_decode_recovers_t_hat(self):
        """decode_public_key recovers the original t_hat."""
        t_hat = self._make_t_hat()
        ek = encode_public_key(t_hat, RHO)
        t_hat2, rho2 = decode_public_key(ek)
        assert t_hat == t_hat2
        assert rho2 == RHO

    def test_decode_wrong_length_raises(self):
        """decode_public_key raises ValueError for wrong length."""
        with pytest.raises(ValueError, match="800"):
            decode_public_key(bytes(100))


# ---------------------------------------------------------------------------
# encode/decode secret key tests
# ---------------------------------------------------------------------------


class TestSecretKeyEncoding:
    def _make_s_hat(self):
        return ntt_vec(sample_secret_vector(SIGMA))

    def test_encoded_length_768(self):
        """Encoded secret key is exactly 768 bytes."""
        s_hat = self._make_s_hat()
        dk = encode_secret_key(s_hat)
        assert len(dk) == 768

    def test_decode_recovers_s_hat(self):
        """decode_secret_key recovers the original s_hat."""
        s_hat = self._make_s_hat()
        dk = encode_secret_key(s_hat)
        s_hat2 = decode_secret_key(dk)
        assert s_hat == s_hat2

    def test_decode_wrong_length_raises(self):
        """decode_secret_key raises ValueError for wrong length."""
        with pytest.raises(ValueError, match="768"):
            decode_secret_key(bytes(100))

    def test_deterministic(self):
        """Same s_hat always encodes to same bytes."""
        s_hat = self._make_s_hat()
        assert encode_secret_key(s_hat) == encode_secret_key(s_hat)
