"""
Milestone 8.3: Negative Tests for ML-KEM-512.

Test 1: Corrupted ciphertext
  - Flip bits at various positions
  - Decaps must produce different K (implicit rejection)
  - Must never crash

Test 2: Wrong key
  - Encaps with one key, Decaps with another
  - Must produce different K

Test 3: Invalid inputs
  - Wrong sizes for all API functions
  - Must raise ValueError gracefully, never crash
"""

import pytest

from ml_kem_512.kem.decaps import decaps
from ml_kem_512.kem.encaps import encaps
from ml_kem_512.kem.keygen import keygen
from ml_kem_512.pke.decrypt import decrypt_pke
from ml_kem_512.pke.encrypt import encrypt_pke
from ml_kem_512.pke.keygen import decode_public_key, decode_secret_key, keygen_pke

# Fixed seeds for reproducibility
D1 = bytes(range(32))
Z1 = bytes(range(1, 33))
D2 = bytes(range(2, 34))
Z2 = bytes(range(3, 35))
MSG = bytes([0x42] * 32)
RAND = bytes([0xAB] * 32)


# ---------------------------------------------------------------------------
# Test 1: Corrupted ciphertext — implicit rejection
# ---------------------------------------------------------------------------


class TestCorruptedCiphertext:
    def setup_method(self):
        self.ek, self.dk = keygen(D1, Z1)
        self.K_valid, self.c = encaps(self.ek, MSG)

    def test_single_bit_flip_triggers_rejection(self):
        """Flipping one bit in ciphertext produces different K."""
        c_bad = bytes([self.c[0] ^ 0x01]) + self.c[1:]
        K_bad = decaps(c_bad, self.dk)
        assert K_bad != self.K_valid

    def test_rejection_at_multiple_positions(self):
        """Corruption at any byte position triggers rejection."""
        positions = [0, 1, 100, 319, 320, 321, 639, 640, 641, 767]
        for pos in positions:
            c_bad = self.c[:pos] + bytes([self.c[pos] ^ 0xFF]) + self.c[pos + 1 :]
            K_bad = decaps(c_bad, self.dk)
            assert K_bad != self.K_valid, f"No rejection at position {pos}"

    def test_rejection_never_crashes(self):
        """Corrupted ciphertext never raises an exception."""
        for pos in range(0, 768, 50):
            c_bad = self.c[:pos] + bytes([self.c[pos] ^ 0x55]) + self.c[pos + 1 :]
            K = decaps(c_bad, self.dk)
            assert len(K) == 32

    def test_rejection_returns_32_bytes(self):
        """Implicit rejection always returns exactly 32 bytes."""
        c_bad = bytes([self.c[0] ^ 0xFF]) + self.c[1:]
        K = decaps(c_bad, self.dk)
        assert len(K) == 32

    def test_rejection_key_not_zero(self):
        """Rejection key is not trivially zero."""
        c_bad = bytes([self.c[0] ^ 0x01]) + self.c[1:]
        K = decaps(c_bad, self.dk)
        assert K != bytes(32)

    def test_rejection_is_deterministic(self):
        """Same corrupted ciphertext always produces same rejection K."""
        c_bad = bytes([self.c[0] ^ 0x01]) + self.c[1:]
        K1 = decaps(c_bad, self.dk)
        K2 = decaps(c_bad, self.dk)
        assert K1 == K2

    def test_different_corruptions_different_rejection_keys(self):
        """Different corruptions produce different rejection keys."""
        c_bad1 = bytes([self.c[0] ^ 0x01]) + self.c[1:]
        c_bad2 = bytes([self.c[0] ^ 0x02]) + self.c[1:]
        K1 = decaps(c_bad1, self.dk)
        K2 = decaps(c_bad2, self.dk)
        assert K1 != K2

    def test_all_zeros_ciphertext(self):
        """All-zero ciphertext triggers rejection without crash."""
        K = decaps(bytes(768), self.dk)
        assert len(K) == 32
        assert K != self.K_valid

    def test_all_ones_ciphertext(self):
        """All-ones ciphertext triggers rejection without crash."""
        K = decaps(bytes([0xFF] * 768), self.dk)
        assert len(K) == 32
        assert K != self.K_valid

    def test_pke_decrypt_corrupted_ciphertext(self):
        """K-PKE.Decrypt on corrupted ciphertext returns 32 bytes (no crash)."""
        ek_pke, dk_pke = keygen_pke(D1)
        c = encrypt_pke(ek_pke, MSG, RAND)
        c_bad = bytes([c[0] ^ 0x01]) + c[1:]
        m_bad = decrypt_pke(dk_pke, c_bad)
        # K-PKE.Decrypt may or may not recover the original message
        # depending on how much the corruption affects the LWE noise.
        # What matters is: no crash and output is 32 bytes.
        assert len(m_bad) == 32


# ---------------------------------------------------------------------------
# Test 2: Wrong key
# ---------------------------------------------------------------------------


class TestWrongKey:
    def setup_method(self):
        self.ek1, self.dk1 = keygen(D1, Z1)
        self.ek2, self.dk2 = keygen(D2, Z2)
        self.K_valid, self.c = encaps(self.ek1, MSG)

    def test_wrong_dk_produces_different_k(self):
        """Decaps with wrong dk produces different K."""
        K_wrong = decaps(self.c, self.dk2)
        assert K_wrong != self.K_valid

    def test_wrong_dk_no_crash(self):
        """Decaps with wrong dk does not crash."""
        K = decaps(self.c, self.dk2)
        assert len(K) == 32

    def test_cross_key_always_fails(self):
        """5 different key pairs: cross-decaps always produces wrong K."""
        keys = [keygen(bytes([i] * 32), bytes([i + 50] * 32)) for i in range(5)]
        for i, (ek_i, dk_i) in enumerate(keys):
            K_enc, c_i = encaps(ek_i, MSG)
            for j, (_, dk_j) in enumerate(keys):
                K_dec = decaps(c_i, dk_j)
                if i == j:
                    assert K_dec == K_enc, f"Correct key failed at i={i}"
                else:
                    assert K_dec != K_enc, f"Wrong key succeeded at i={i}, j={j}"

    def test_pke_wrong_secret_key(self):
        """K-PKE.Decrypt with wrong secret key returns wrong message."""
        ek1, dk1 = keygen_pke(D1)
        _, dk2 = keygen_pke(D2)
        c = encrypt_pke(ek1, MSG, RAND)
        m_wrong = decrypt_pke(dk2, c)
        assert m_wrong != MSG
        assert len(m_wrong) == 32


# ---------------------------------------------------------------------------
# Test 3: Invalid inputs — graceful error handling
# ---------------------------------------------------------------------------


class TestInvalidInputs:
    def setup_method(self):
        self.ek, self.dk = keygen(D1, Z1)
        _, self.c = encaps(self.ek, MSG)

    # keygen
    def test_keygen_short_d(self):
        with pytest.raises(ValueError):
            keygen(bytes(16), Z1)

    def test_keygen_long_d(self):
        with pytest.raises(ValueError):
            keygen(bytes(64), Z1)

    def test_keygen_short_z(self):
        with pytest.raises(ValueError):
            keygen(D1, bytes(16))

    def test_keygen_long_z(self):
        with pytest.raises(ValueError):
            keygen(D1, bytes(64))

    # encaps
    def test_encaps_short_ek(self):
        with pytest.raises(ValueError):
            encaps(bytes(100), MSG)

    def test_encaps_long_ek(self):
        with pytest.raises(ValueError):
            encaps(bytes(1000), MSG)

    def test_encaps_short_m(self):
        with pytest.raises(ValueError):
            encaps(self.ek, bytes(16))

    def test_encaps_long_m(self):
        with pytest.raises(ValueError):
            encaps(self.ek, bytes(64))

    # decaps
    def test_decaps_short_c(self):
        with pytest.raises(ValueError):
            decaps(bytes(100), self.dk)

    def test_decaps_long_c(self):
        with pytest.raises(ValueError):
            decaps(bytes(1000), self.dk)

    def test_decaps_short_dk(self):
        with pytest.raises(ValueError):
            decaps(self.c, bytes(100))

    def test_decaps_long_dk(self):
        with pytest.raises(ValueError):
            decaps(self.c, bytes(2000))

    # keygen_pke
    def test_keygen_pke_short_d(self):
        with pytest.raises(ValueError):
            keygen_pke(bytes(16))

    def test_keygen_pke_long_d(self):
        with pytest.raises(ValueError):
            keygen_pke(bytes(64))

    # encrypt_pke
    def test_encrypt_pke_short_m(self):
        with pytest.raises(ValueError):
            encrypt_pke(self.ek, bytes(16), RAND)

    def test_encrypt_pke_long_m(self):
        with pytest.raises(ValueError):
            encrypt_pke(self.ek, bytes(64), RAND)

    def test_encrypt_pke_short_r(self):
        with pytest.raises(ValueError):
            encrypt_pke(self.ek, MSG, bytes(16))

    def test_encrypt_pke_long_r(self):
        with pytest.raises(ValueError):
            encrypt_pke(self.ek, MSG, bytes(64))

    # decode_public_key / decode_secret_key
    def test_decode_public_key_wrong_size(self):
        with pytest.raises(ValueError, match="800"):
            decode_public_key(bytes(100))

    def test_decode_secret_key_wrong_size(self):
        with pytest.raises(ValueError, match="768"):
            decode_secret_key(bytes(100))

    # empty inputs
    def test_encaps_empty_ek(self):
        with pytest.raises(ValueError):
            encaps(b"", MSG)

    def test_decaps_empty_c(self):
        with pytest.raises(ValueError):
            decaps(b"", self.dk)

    def test_decaps_empty_dk(self):
        with pytest.raises(ValueError):
            decaps(self.c, b"")
