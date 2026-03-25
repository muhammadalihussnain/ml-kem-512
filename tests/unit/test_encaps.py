"""
Unit tests for Phase 6 — Encapsulation (Milestones 6.1, 6.2, 6.3).

Milestone 6.1 — Message encoding / KDF derivation:
  - m is 32 bytes
  - (K_bar, r) = G(m || H(ek)) produces 32+32 bytes

Milestone 6.2 — K-PKE.Encrypt:
  - Ciphertext is 768 bytes
  - Deterministic from (ek, m, r)
  - u and v computed correctly
  - decode_ciphertext recovers (u, v)

Milestone 6.3 — ML-KEM.Encaps:
  - K is 32 bytes
  - c is 768 bytes
  - Deterministic from (ek, m)
  - Different m -> different (K, c)
  - keygen + encaps produces valid outputs
"""

import pytest

from ml_kem_512.kem.encaps import encaps
from ml_kem_512.kem.keygen import keygen
from ml_kem_512.pke.encrypt import decode_ciphertext, encrypt_pke
from ml_kem_512.pke.keygen import keygen_pke
from ml_kem_512.polynomial.poly import N, Q
from ml_kem_512.primitives.kdf import G, H, J

SEED_D = bytes(range(32))
SEED_Z = bytes(range(1, 33))
MSG = bytes([0x42] * 32)
RAND = bytes([0xAB] * 32)


# ---------------------------------------------------------------------------
# Milestone 6.1: Message encoding / KDF derivation
# ---------------------------------------------------------------------------


class TestMessageEncoding:
    def test_h_ek_is_32_bytes(self):
        """H(ek) produces 32 bytes."""
        ek, _ = keygen_pke(SEED_D)
        assert len(H(ek)) == 32

    def test_g_produces_k_bar_and_r(self):
        """G(m || H(ek)) produces two 32-byte values."""
        ek, _ = keygen_pke(SEED_D)
        k_bar, r = G(MSG + H(ek))
        assert len(k_bar) == 32
        assert len(r) == 32

    def test_k_bar_and_r_differ(self):
        """K_bar and r are different."""
        ek, _ = keygen_pke(SEED_D)
        k_bar, r = G(MSG + H(ek))
        assert k_bar != r

    def test_different_m_different_r(self):
        """Different m produces different r."""
        ek, _ = keygen_pke(SEED_D)
        _, r1 = G(MSG + H(ek))
        _, r2 = G(bytes([0x00] * 32) + H(ek))
        assert r1 != r2

    def test_deterministic_derivation(self):
        """Same (m, ek) always produces same (K_bar, r)."""
        ek, _ = keygen_pke(SEED_D)
        result1 = G(MSG + H(ek))
        result2 = G(MSG + H(ek))
        assert result1 == result2


# ---------------------------------------------------------------------------
# Milestone 6.2: K-PKE.Encrypt
# ---------------------------------------------------------------------------


class TestEncryptPKE:
    def setup_method(self):
        self.ek, self.dk = keygen_pke(SEED_D)

    def test_ciphertext_size_768(self):
        """Ciphertext is exactly 768 bytes."""
        c = encrypt_pke(self.ek, MSG, RAND)
        assert len(c) == 768

    def test_c1_size_640(self):
        """First part c1 (u encoding) is 640 bytes."""
        c = encrypt_pke(self.ek, MSG, RAND)
        assert len(c[:640]) == 640

    def test_c2_size_128(self):
        """Second part c2 (v encoding) is 128 bytes."""
        c = encrypt_pke(self.ek, MSG, RAND)
        assert len(c[640:]) == 128

    def test_deterministic(self):
        """Same (ek, m, r) always produces same ciphertext."""
        c1 = encrypt_pke(self.ek, MSG, RAND)
        c2 = encrypt_pke(self.ek, MSG, RAND)
        assert c1 == c2

    def test_different_r_different_ciphertext(self):
        """Different r produces different ciphertext."""
        c1 = encrypt_pke(self.ek, MSG, RAND)
        c2 = encrypt_pke(self.ek, MSG, bytes([0x00] * 32))
        assert c1 != c2

    def test_different_m_different_ciphertext(self):
        """Different m produces different ciphertext."""
        c1 = encrypt_pke(self.ek, MSG, RAND)
        c2 = encrypt_pke(self.ek, bytes([0x00] * 32), RAND)
        assert c1 != c2

    def test_invalid_m_length_raises(self):
        """Wrong message length raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            encrypt_pke(self.ek, bytes(16), RAND)

    def test_invalid_r_length_raises(self):
        """Wrong randomness length raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            encrypt_pke(self.ek, MSG, bytes(16))

    def test_decode_ciphertext_size(self):
        """decode_ciphertext accepts 768-byte ciphertext."""
        c = encrypt_pke(self.ek, MSG, RAND)
        u, v = decode_ciphertext(c)
        assert len(u) == 2  # k=2 polynomials
        assert len(v.coeffs) == N

    def test_decode_ciphertext_coefficients_in_range(self):
        """Decoded u and v have coefficients in [0, Q-1]."""
        c = encrypt_pke(self.ek, MSG, RAND)
        u, v = decode_ciphertext(c)
        for i in range(2):
            assert all(0 <= coef < Q for coef in u[i].coeffs)
        assert all(0 <= coef < Q for coef in v.coeffs)

    def test_decode_wrong_size_raises(self):
        """decode_ciphertext raises ValueError for wrong size."""
        with pytest.raises(ValueError, match="768"):
            decode_ciphertext(bytes(100))


# ---------------------------------------------------------------------------
# Milestone 6.3: ML-KEM.Encaps
# ---------------------------------------------------------------------------


class TestEncaps:
    def setup_method(self):
        self.ek, self.dk = keygen(SEED_D, SEED_Z)

    def test_shared_secret_32_bytes(self):
        """Shared secret K is exactly 32 bytes."""
        K, c = encaps(self.ek, MSG)
        assert len(K) == 32

    def test_ciphertext_768_bytes(self):
        """Ciphertext c is exactly 768 bytes."""
        K, c = encaps(self.ek, MSG)
        assert len(c) == 768

    def test_deterministic_from_m(self):
        """Same (ek, m) always produces same (K, c)."""
        K1, c1 = encaps(self.ek, MSG)
        K2, c2 = encaps(self.ek, MSG)
        assert K1 == K2
        assert c1 == c2

    def test_different_m_different_output(self):
        """Different m produces different K and c."""
        K1, c1 = encaps(self.ek, MSG)
        K2, c2 = encaps(self.ek, bytes([0x00] * 32))
        assert K1 != K2
        assert c1 != c2

    def test_invalid_ek_size_raises(self):
        """Wrong ek size raises ValueError."""
        with pytest.raises(ValueError, match="800"):
            encaps(bytes(100), MSG)

    def test_invalid_m_size_raises(self):
        """Wrong m size raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            encaps(self.ek, bytes(16))

    def test_invalid_m_too_long_raises(self):
        """m longer than 32 bytes raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            encaps(self.ek, bytes(64))

    def test_k_is_bytes(self):
        """Shared secret K is bytes."""
        K, c = encaps(self.ek, MSG)
        assert isinstance(K, bytes)
        assert isinstance(c, bytes)

    def test_k_not_all_zeros(self):
        """Shared secret K is not trivially zero."""
        K, c = encaps(self.ek, MSG)
        assert K != bytes(32)


# ---------------------------------------------------------------------------
# ML-KEM.KeyGen tests (Phase 6 prerequisite)
# ---------------------------------------------------------------------------


class TestMLKEMKeyGen:
    def test_ek_size_800(self):
        """Encapsulation key is 800 bytes."""
        ek, dk = keygen(SEED_D, SEED_Z)
        assert len(ek) == 800

    def test_dk_size_1632(self):
        """Decapsulation key is 1632 bytes."""
        ek, dk = keygen(SEED_D, SEED_Z)
        assert len(dk) == 1632

    def test_dk_structure(self):
        """
        dk = dk_pke(768) || ek(800) || H(ek)(32) || z(32) = 1632 bytes.
        """
        ek, dk = keygen(SEED_D, SEED_Z)
        assert dk[768 : 768 + 800] == ek
        assert dk[768 + 800 : 768 + 800 + 32] == H(ek)
        assert dk[1600:] == SEED_Z

    def test_deterministic(self):
        """Same (d, z) always produces same (ek, dk)."""
        ek1, dk1 = keygen(SEED_D, SEED_Z)
        ek2, dk2 = keygen(SEED_D, SEED_Z)
        assert ek1 == ek2
        assert dk1 == dk2

    def test_different_seeds_different_keys(self):
        """Different seeds produce different keys."""
        ek1, _ = keygen(bytes(32), SEED_Z)
        ek2, _ = keygen(bytes(range(32)), SEED_Z)
        assert ek1 != ek2

    def test_invalid_d_raises(self):
        """Wrong d length raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            keygen(bytes(16), SEED_Z)

    def test_invalid_z_raises(self):
        """Wrong z length raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            keygen(SEED_D, bytes(16))

    def test_random_keygen(self):
        """keygen() with no args generates random keys of correct size."""
        ek, dk = keygen()
        assert len(ek) == 800
        assert len(dk) == 1632

    def test_encaps_random_m(self):
        """encaps(ek) with no m generates random message and returns valid output."""
        ek, _ = keygen(SEED_D, SEED_Z)
        K, c = encaps(ek)
        assert len(K) == 32
        assert len(c) == 768
