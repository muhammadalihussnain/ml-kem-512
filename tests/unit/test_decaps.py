"""
Unit tests for Phase 7 — Decapsulation (Milestones 7.1 & 7.2).

Milestone 7.1 — K-PKE.Decrypt:
  - Decrypt(Encrypt(m)) == m
  - Output is 32 bytes
  - Works with valid ciphertext

Milestone 7.2 — ML-KEM.Decaps:
  - Decaps(Encaps(ek)) produces same shared secret
  - Corrupted ciphertext triggers implicit rejection (different K)
  - Wrong key produces different K
  - dk parsing is correct
  - Constant-time equality check
"""

import pytest

from ml_kem_512.kem.decaps import _ct_eq, _parse_dk, decaps
from ml_kem_512.kem.encaps import encaps
from ml_kem_512.kem.keygen import keygen
from ml_kem_512.pke.decrypt import decrypt_pke
from ml_kem_512.pke.encrypt import encrypt_pke
from ml_kem_512.pke.keygen import keygen_pke
from ml_kem_512.primitives.kdf import H

SEED_D = bytes(range(32))
SEED_Z = bytes(range(1, 33))
MSG = bytes([0x42] * 32)
RAND = bytes([0xAB] * 32)


# ---------------------------------------------------------------------------
# Milestone 7.1: K-PKE.Decrypt
# ---------------------------------------------------------------------------


class TestDecryptPKE:
    def setup_method(self):
        self.ek, self.dk_pke = keygen_pke(SEED_D)

    def test_decrypt_returns_32_bytes(self):
        """decrypt_pke returns exactly 32 bytes."""
        c = encrypt_pke(self.ek, MSG, RAND)
        m = decrypt_pke(self.dk_pke, c)
        assert len(m) == 32

    def test_decrypt_recovers_message(self):
        """Decrypt(Encrypt(m)) == m."""
        c = encrypt_pke(self.ek, MSG, RAND)
        m_recovered = decrypt_pke(self.dk_pke, c)
        assert m_recovered == MSG

    def test_decrypt_different_messages(self):
        """Different messages decrypt to different values."""
        m1 = bytes([0x00] * 32)
        m2 = bytes([0xFF] * 32)
        c1 = encrypt_pke(self.ek, m1, RAND)
        c2 = encrypt_pke(self.ek, m2, RAND)
        assert decrypt_pke(self.dk_pke, c1) == m1
        assert decrypt_pke(self.dk_pke, c2) == m2

    def test_decrypt_deterministic(self):
        """Same (dk, c) always produces same m."""
        c = encrypt_pke(self.ek, MSG, RAND)
        assert decrypt_pke(self.dk_pke, c) == decrypt_pke(self.dk_pke, c)

    def test_decrypt_all_zero_message(self):
        """All-zero message encrypts and decrypts correctly."""
        m = bytes(32)
        c = encrypt_pke(self.ek, m, RAND)
        assert decrypt_pke(self.dk_pke, c) == m

    def test_decrypt_all_ones_message(self):
        """All-ones message encrypts and decrypts correctly."""
        m = bytes([0xFF] * 32)
        c = encrypt_pke(self.ek, m, RAND)
        assert decrypt_pke(self.dk_pke, c) == m

    def test_decrypt_multiple_messages(self):
        """10 different messages all decrypt correctly."""
        for i in range(10):
            m = bytes([i] * 32)
            c = encrypt_pke(self.ek, m, RAND)
            assert decrypt_pke(self.dk_pke, c) == m


# ---------------------------------------------------------------------------
# Milestone 7.2: ML-KEM.Decaps
# ---------------------------------------------------------------------------


class TestDecaps:
    def setup_method(self):
        self.ek, self.dk = keygen(SEED_D, SEED_Z)

    def test_decaps_returns_32_bytes(self):
        """Decaps returns exactly 32 bytes."""
        _, c = encaps(self.ek, MSG)
        K = decaps(c, self.dk)
        assert len(K) == 32

    def test_encaps_decaps_shared_secret_matches(self):
        """Encaps and Decaps produce the same shared secret."""
        K_enc, c = encaps(self.ek, MSG)
        K_dec = decaps(c, self.dk)
        assert K_enc == K_dec

    def test_shared_secret_not_zero(self):
        """Shared secret is not trivially zero."""
        K_enc, c = encaps(self.ek, MSG)
        K_dec = decaps(c, self.dk)
        assert K_dec != bytes(32)

    def test_multiple_encaps_decaps(self):
        """10 different messages all produce matching shared secrets."""
        for i in range(10):
            m = bytes([i] * 32)
            K_enc, c = encaps(self.ek, m)
            K_dec = decaps(c, self.dk)
            assert K_enc == K_dec

    def test_corrupted_ciphertext_implicit_rejection(self):
        """
        Corrupted ciphertext triggers implicit rejection.
        Decaps returns a different K than the original.
        """
        K_enc, c = encaps(self.ek, MSG)
        # Flip a bit in the ciphertext
        c_corrupted = bytes([c[0] ^ 0x01]) + c[1:]
        K_dec = decaps(c_corrupted, self.dk)
        assert K_dec != K_enc

    def test_corrupted_ciphertext_still_returns_32_bytes(self):
        """Corrupted ciphertext still returns 32 bytes (no crash)."""
        _, c = encaps(self.ek, MSG)
        c_corrupted = bytes([c[100] ^ 0xFF]) + c[1:]
        K = decaps(c_corrupted, self.dk)
        assert len(K) == 32

    def test_wrong_key_different_shared_secret(self):
        """Using a different key pair produces a different shared secret."""
        K_enc, c = encaps(self.ek, MSG)
        # Generate a different key pair
        ek2, dk2 = keygen(bytes(range(1, 33)), SEED_Z)
        K_wrong = decaps(c, dk2)
        assert K_wrong != K_enc

    def test_invalid_ciphertext_size_raises(self):
        """Wrong ciphertext size raises ValueError."""
        with pytest.raises(ValueError, match="768"):
            decaps(bytes(100), self.dk)

    def test_invalid_dk_size_raises(self):
        """Wrong dk size raises ValueError."""
        _, c = encaps(self.ek, MSG)
        with pytest.raises(ValueError, match="1632"):
            decaps(c, bytes(100))

    def test_rejection_key_is_deterministic(self):
        """
        Same corrupted ciphertext always produces same rejection K.
        (J(z || H(c)) is deterministic)
        """
        _, c = encaps(self.ek, MSG)
        c_bad = bytes([c[0] ^ 0x01]) + c[1:]
        K1 = decaps(c_bad, self.dk)
        K2 = decaps(c_bad, self.dk)
        assert K1 == K2

    def test_different_corruptions_different_rejection_keys(self):
        """Different corrupted ciphertexts produce different rejection keys."""
        _, c = encaps(self.ek, MSG)
        c_bad1 = bytes([c[0] ^ 0x01]) + c[1:]
        c_bad2 = bytes([c[0] ^ 0x02]) + c[1:]
        K1 = decaps(c_bad1, self.dk)
        K2 = decaps(c_bad2, self.dk)
        assert K1 != K2


# ---------------------------------------------------------------------------
# dk parsing tests
# ---------------------------------------------------------------------------


class TestParseDK:
    def test_parse_dk_correct_lengths(self):
        """_parse_dk returns components of correct lengths."""
        _, dk = keygen(SEED_D, SEED_Z)
        dk_pke, ek, h, z = _parse_dk(dk)
        assert len(dk_pke) == 768
        assert len(ek) == 800
        assert len(h) == 32
        assert len(z) == 32

    def test_parse_dk_h_is_h_ek(self):
        """h component equals H(ek)."""
        ek, dk = keygen(SEED_D, SEED_Z)
        _, ek_parsed, h, _ = _parse_dk(dk)
        assert h == H(ek_parsed)
        assert ek_parsed == ek

    def test_parse_dk_z_is_seed(self):
        """z component equals the z seed used in keygen."""
        _, dk = keygen(SEED_D, SEED_Z)
        _, _, _, z = _parse_dk(dk)
        assert z == SEED_Z

    def test_parse_dk_wrong_size_raises(self):
        """_parse_dk raises ValueError for wrong size."""
        with pytest.raises(ValueError, match="1632"):
            _parse_dk(bytes(100))


# ---------------------------------------------------------------------------
# Constant-time equality tests
# ---------------------------------------------------------------------------


class TestConstantTimeEq:
    def test_equal_bytes(self):
        """_ct_eq returns True for equal byte strings."""
        assert _ct_eq(b"hello", b"hello")

    def test_unequal_bytes(self):
        """_ct_eq returns False for unequal byte strings."""
        assert not _ct_eq(b"hello", b"world")

    def test_different_lengths(self):
        """_ct_eq returns False for different lengths."""
        assert not _ct_eq(b"abc", b"abcd")

    def test_empty_bytes(self):
        """_ct_eq returns True for two empty byte strings."""
        assert _ct_eq(b"", b"")

    def test_single_bit_difference(self):
        """_ct_eq returns False when only one bit differs."""
        a = bytes([0x00] * 768)
        b = bytes([0x01]) + bytes([0x00] * 767)
        assert not _ct_eq(a, b)

    def test_ciphertext_self_equal(self):
        """A ciphertext is equal to itself."""
        _, c = encaps(keygen(SEED_D, SEED_Z)[0], MSG)
        assert _ct_eq(c, c)
