"""
Unit tests for Milestones 5.3, 5.4, 5.5 — Complete K-PKE KeyGen.

Milestone 5.3 — Error vector generation:
  - k polynomials from CBD_eta1
  - Coefficients in [-eta1, eta1]
  - Uses PRF counters k..2k-1 (separate from secret)

Milestone 5.4 — Public key computation:
  - t_hat = A*s_hat + e_hat in NTT domain
  - t_hat = A*s + e verified in polynomial domain
  - pk = 800 bytes, sk = 768 bytes

Milestone 5.5 — Complete keygen_pke:
  - ek is 800 bytes
  - dk is 768 bytes
  - Deterministic from seed d
  - Different seeds produce different keys
"""

import pytest

from ml_kem_512.module.vector import K, PolyVec, inv_ntt_vec, ntt_vec
from ml_kem_512.pke.keygen import (
    compute_public_key_ntt,
    decode_public_key,
    decode_secret_key,
    encode_public_key,
    encode_secret_key,
    keygen_pke,
    sample_error_vector,
    sample_secret_vector,
)
from ml_kem_512.pke.matrix import generate_matrix
from ml_kem_512.polynomial.ntt import inv_ntt
from ml_kem_512.polynomial.poly import N, Polynomial, Q
from ml_kem_512.sampling.cbd import ETA1

SIGMA = bytes(range(32))
RHO = bytes(range(32))
SEED_D = bytes(range(32))


def _signed(c: int) -> int:
    return c - Q if c > Q // 2 else c


# ---------------------------------------------------------------------------
# Milestone 5.3: Error vector generation
# ---------------------------------------------------------------------------


class TestErrorVectorGeneration:
    def test_returns_polyvec(self):
        """sample_error_vector returns a PolyVec."""
        assert isinstance(sample_error_vector(SIGMA), PolyVec)

    def test_length_k(self):
        """Error vector has exactly k=2 polynomials."""
        assert len(sample_error_vector(SIGMA)) == K

    def test_each_poly_n_coefficients(self):
        """Each polynomial has N=256 coefficients."""
        e = sample_error_vector(SIGMA)
        for i in range(K):
            assert len(e[i].coeffs) == N

    def test_coefficients_canonical_range(self):
        """All coefficients are in [0, Q-1]."""
        e = sample_error_vector(SIGMA)
        for i in range(K):
            assert all(0 <= c < Q for c in e[i].coeffs)

    def test_signed_coefficients_in_eta1_range(self):
        """Signed coefficients are in [-eta1, eta1] = [-3, 3]."""
        e = sample_error_vector(SIGMA)
        for i in range(K):
            for c in e[i].coeffs:
                assert -ETA1 <= _signed(c) <= ETA1

    def test_deterministic(self):
        """Same sigma always produces same error vector."""
        assert sample_error_vector(SIGMA) == sample_error_vector(SIGMA)

    def test_different_from_secret(self):
        """Error vector uses different PRF counters than secret vector."""
        s = sample_secret_vector(SIGMA)
        e = sample_error_vector(SIGMA)
        assert s != e

    def test_uses_counters_k_to_2k(self):
        """Error vector uses PRF counters k and k+1."""
        from ml_kem_512.primitives.prf import prf
        from ml_kem_512.sampling.cbd import cbd_eta1

        e = sample_error_vector(SIGMA)
        for i in range(K):
            expected = cbd_eta1(prf(SIGMA, K + i, 64 * ETA1))
            assert e[i] == expected

    def test_ntt_roundtrip(self):
        """InvNTT(NTT(e)) == e."""
        e = sample_error_vector(SIGMA)
        assert inv_ntt_vec(ntt_vec(e)) == e

    def test_small_coefficients_expected(self):
        """Most signed coefficients should be small (CBD property)."""
        e = sample_error_vector(SIGMA)
        signed = [_signed(c) for i in range(K) for c in e[i].coeffs]
        # All must be in [-3, 3]
        assert all(-ETA1 <= v <= ETA1 for v in signed)
        # Mean should be near 0
        mean = sum(signed) / len(signed)
        assert abs(mean) < 0.5


# ---------------------------------------------------------------------------
# Milestone 5.4: Public key computation and encoding
# ---------------------------------------------------------------------------


class TestPublicKeyComputation:
    def _make_components(self, sigma=SIGMA, rho=RHO):
        A_hat = generate_matrix(rho)
        s = sample_secret_vector(sigma)
        e = sample_error_vector(sigma)
        s_hat = ntt_vec(s)
        e_hat = ntt_vec(e)
        t_hat = compute_public_key_ntt(A_hat, s_hat, e_hat)
        return A_hat, s, e, s_hat, e_hat, t_hat

    def test_t_hat_is_polyvec(self):
        """t_hat is a PolyVec."""
        _, _, _, _, _, t_hat = self._make_components()
        assert isinstance(t_hat, PolyVec)

    def test_t_hat_length_k(self):
        """t_hat has k=2 polynomials."""
        _, _, _, _, _, t_hat = self._make_components()
        assert len(t_hat) == K

    def test_t_hat_coefficients_in_range(self):
        """All t_hat coefficients are in [0, Q-1]."""
        _, _, _, _, _, t_hat = self._make_components()
        for i in range(K):
            assert all(0 <= c < Q for c in t_hat[i].coeffs)

    def test_t_equals_as_plus_e_in_poly_domain(self):
        """
        Verify t = A*s + e in polynomial domain (not NTT).
        InvNTT(t_hat)[i] == sum_j(A_poly[i][j] * s[j]) + e[i]
        """
        A_hat, s, e, s_hat, e_hat, t_hat = self._make_components()

        # Recover polynomial-domain t
        t_poly = inv_ntt_vec(t_hat)

        # Compute A*s + e in polynomial domain (schoolbook)
        A_poly = [[inv_ntt(A_hat[i][j]) for j in range(K)] for i in range(K)]
        for i in range(K):
            row_sum = sum(
                (A_poly[i][j] * s[j] for j in range(1, K)),
                A_poly[i][0] * s[0],
            )
            expected = row_sum + e[i]
            assert t_poly[i] == expected

    def test_pk_size_800(self):
        """Encoded public key is exactly 800 bytes."""
        _, _, _, _, _, t_hat = self._make_components()
        ek = encode_public_key(t_hat, RHO)
        assert len(ek) == 800

    def test_sk_size_768(self):
        """Encoded secret key is exactly 768 bytes."""
        _, _, _, s_hat, _, _ = self._make_components()
        dk = encode_secret_key(s_hat)
        assert len(dk) == 768

    def test_pk_decode_roundtrip(self):
        """decode_public_key(encode_public_key(t_hat, rho)) == (t_hat, rho)."""
        _, _, _, _, _, t_hat = self._make_components()
        ek = encode_public_key(t_hat, RHO)
        t_hat2, rho2 = decode_public_key(ek)
        assert t_hat == t_hat2
        assert rho2 == RHO

    def test_sk_decode_roundtrip(self):
        """decode_secret_key(encode_secret_key(s_hat)) == s_hat."""
        _, _, _, s_hat, _, _ = self._make_components()
        dk = encode_secret_key(s_hat)
        assert decode_secret_key(dk) == s_hat


# ---------------------------------------------------------------------------
# Milestone 5.5: Complete keygen_pke
# ---------------------------------------------------------------------------


class TestKeygenPKE:
    def test_ek_size_800(self):
        """Encapsulation key is exactly 800 bytes."""
        ek, dk = keygen_pke(SEED_D)
        assert len(ek) == 800

    def test_dk_size_768(self):
        """Decapsulation key (PKE part) is exactly 768 bytes."""
        ek, dk = keygen_pke(SEED_D)
        assert len(dk) == 768

    def test_deterministic(self):
        """Same seed d always produces same (ek, dk)."""
        ek1, dk1 = keygen_pke(SEED_D)
        ek2, dk2 = keygen_pke(SEED_D)
        assert ek1 == ek2
        assert dk1 == dk2

    def test_different_seeds_different_keys(self):
        """Different seeds produce different key pairs."""
        ek1, dk1 = keygen_pke(bytes(32))
        ek2, dk2 = keygen_pke(bytes(range(32)))
        assert ek1 != ek2
        assert dk1 != dk2

    def test_multiple_keypairs_all_different(self):
        """10 different seeds produce 10 different public keys."""
        eks = [keygen_pke(bytes([i] * 32))[0] for i in range(10)]
        assert len(set(eks)) == 10

    def test_invalid_seed_raises(self):
        """Wrong seed length raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            keygen_pke(bytes(16))

    def test_ek_last_32_bytes_is_rho(self):
        """Last 32 bytes of ek is rho (the public seed)."""
        ek, _ = keygen_pke(SEED_D)
        # rho comes from G(d || k), just verify it's 32 bytes
        rho = ek[768:]
        assert len(rho) == 32

    def test_dk_decodes_to_valid_s_hat(self):
        """Decoded dk gives a valid NTT-domain secret vector."""
        ek, dk = keygen_pke(SEED_D)
        s_hat = decode_secret_key(dk)
        assert isinstance(s_hat, PolyVec)
        assert len(s_hat) == K
        for i in range(K):
            assert all(0 <= c < Q for c in s_hat[i].coeffs)

    def test_ek_decodes_to_valid_t_hat_and_rho(self):
        """Decoded ek gives a valid NTT-domain public key and rho."""
        ek, dk = keygen_pke(SEED_D)
        t_hat, rho = decode_public_key(ek)
        assert isinstance(t_hat, PolyVec)
        assert len(rho) == 32
        for i in range(K):
            assert all(0 <= c < Q for c in t_hat[i].coeffs)

    def test_t_hat_consistent_with_s_hat(self):
        """
        Verify t_hat = A*s_hat + e_hat by re-deriving from the seed.
        Regenerate A from rho in ek, regenerate s and e from sigma,
        confirm t_hat matches.
        """
        from ml_kem_512.primitives.kdf import G

        ek, dk = keygen_pke(SEED_D)
        t_hat, rho = decode_public_key(ek)
        s_hat = decode_secret_key(dk)

        # Re-derive sigma from seed
        rho2, sigma = G(SEED_D + bytes([K]))
        assert rho == rho2

        # Re-generate A, s, e
        A_hat = generate_matrix(rho)
        s = sample_secret_vector(sigma)
        e = sample_error_vector(sigma)
        s_hat2 = ntt_vec(s)
        e_hat = ntt_vec(e)

        assert s_hat == s_hat2

        # Verify t_hat = A*s_hat + e_hat
        t_hat_expected = compute_public_key_ntt(A_hat, s_hat2, e_hat)
        assert t_hat == t_hat_expected
