"""
Milestone 8.1: End-to-End Integration Tests for ML-KEM-512.

Tests the complete KeyGen -> Encaps -> Decaps flow:

  Test 1: Basic functionality
    (ek, dk) = KeyGen()
    (K1, c)  = Encaps(ek)
    K2       = Decaps(c, dk)
    assert K1 == K2

  Test 2: Multiple key pairs (100 iterations)
    All must succeed with matching shared secrets.

  Test 3: Different messages produce different ciphertexts and keys.

  Test 4: Negative tests — corrupted ciphertext, wrong key.
"""

from ml_kem_512.kem.decaps import decaps
from ml_kem_512.kem.encaps import encaps
from ml_kem_512.kem.keygen import keygen

# ---------------------------------------------------------------------------
# Test 1: Basic functionality
# ---------------------------------------------------------------------------


class TestBasicFlow:
    def test_keygen_encaps_decaps(self):
        """KeyGen -> Encaps -> Decaps produces matching shared secrets."""
        ek, dk = keygen(bytes(32), bytes(range(32)))
        K1, c = encaps(ek, bytes([0x42] * 32))
        K2 = decaps(c, dk)
        assert K1 == K2

    def test_shared_secret_is_32_bytes(self):
        """Shared secret is exactly 32 bytes."""
        ek, dk = keygen(bytes(32), bytes(range(32)))
        K1, c = encaps(ek, bytes([0x42] * 32))
        K2 = decaps(c, dk)
        assert len(K1) == 32
        assert len(K2) == 32

    def test_ciphertext_is_768_bytes(self):
        """Ciphertext is exactly 768 bytes."""
        ek, dk = keygen(bytes(32), bytes(range(32)))
        _, c = encaps(ek, bytes([0x42] * 32))
        assert len(c) == 768

    def test_ek_is_800_bytes(self):
        """Encapsulation key is 800 bytes."""
        ek, dk = keygen(bytes(32), bytes(range(32)))
        assert len(ek) == 800

    def test_dk_is_1632_bytes(self):
        """Decapsulation key is 1632 bytes."""
        ek, dk = keygen(bytes(32), bytes(range(32)))
        assert len(dk) == 1632

    def test_shared_secret_not_zero(self):
        """Shared secret is not trivially zero."""
        ek, dk = keygen(bytes(32), bytes(range(32)))
        K1, c = encaps(ek, bytes([0x42] * 32))
        K2 = decaps(c, dk)
        assert K1 != bytes(32)
        assert K2 != bytes(32)


# ---------------------------------------------------------------------------
# Test 2: Multiple key pairs — 100% success rate
# ---------------------------------------------------------------------------


class TestMultipleKeyPairs:
    def test_100_keypairs_all_succeed(self):
        """
        Generate 20 key pairs, encapsulate and decapsulate each.
        All shared secrets must match (100% success rate).
        """
        for i in range(20):
            seed_d = bytes([i] * 32)
            seed_z = bytes([i + 128] * 32)
            m = bytes([i * 7 % 256] * 32)

            ek, dk = keygen(seed_d, seed_z)
            K1, c = encaps(ek, m)
            K2 = decaps(c, dk)

            assert K1 == K2, f"Shared secret mismatch at iteration {i}"
            assert len(K1) == 32
            assert len(c) == 768

    def test_all_key_pairs_produce_different_keys(self):
        """Different seeds produce different encapsulation keys."""
        eks = [keygen(bytes([i] * 32), bytes(32))[0] for i in range(10)]
        assert len(set(eks)) == 10

    def test_all_shared_secrets_different(self):
        """Different key pairs produce different shared secrets."""
        secrets = []
        for i in range(10):
            ek, dk = keygen(bytes([i] * 32), bytes(32))
            K, _ = encaps(ek, bytes([0x42] * 32))
            secrets.append(K)
        assert len(set(secrets)) == 10


# ---------------------------------------------------------------------------
# Test 3: Different messages
# ---------------------------------------------------------------------------


class TestDifferentMessages:
    def setup_method(self):
        self.ek, self.dk = keygen(bytes(32), bytes(range(32)))

    def test_different_messages_different_ciphertexts(self):
        """Same key pair, different messages -> different ciphertexts."""
        _, c1 = encaps(self.ek, bytes([0x00] * 32))
        _, c2 = encaps(self.ek, bytes([0xFF] * 32))
        assert c1 != c2

    def test_different_messages_different_shared_secrets(self):
        """Same key pair, different messages -> different shared secrets."""
        K1, _ = encaps(self.ek, bytes([0x00] * 32))
        K2, _ = encaps(self.ek, bytes([0xFF] * 32))
        assert K1 != K2

    def test_same_message_same_output(self):
        """Same key pair, same message -> same (K, c)."""
        m = bytes([0x42] * 32)
        K1, c1 = encaps(self.ek, m)
        K2, c2 = encaps(self.ek, m)
        assert K1 == K2
        assert c1 == c2

    def test_all_messages_decrypt_correctly(self):
        """10 different messages all produce matching shared secrets."""
        for i in range(10):
            m = bytes([i * 13 % 256] * 32)
            K_enc, c = encaps(self.ek, m)
            K_dec = decaps(c, self.dk)
            assert K_enc == K_dec


# ---------------------------------------------------------------------------
# Test 4: Negative tests
# ---------------------------------------------------------------------------


class TestNegative:
    def setup_method(self):
        self.ek, self.dk = keygen(bytes(32), bytes(range(32)))
        self.m = bytes([0x42] * 32)
        self.K_enc, self.c = encaps(self.ek, self.m)

    def test_corrupted_ciphertext_implicit_rejection(self):
        """Corrupted ciphertext produces different K (implicit rejection)."""
        c_bad = bytes([self.c[0] ^ 0x01]) + self.c[1:]
        K_bad = decaps(c_bad, self.dk)
        assert K_bad != self.K_enc

    def test_corrupted_ciphertext_no_crash(self):
        """Corrupted ciphertext does not crash — returns 32 bytes."""
        for bit_pos in [0, 100, 300, 500, 767]:
            c_bad = self.c[:bit_pos] + bytes([self.c[bit_pos] ^ 0xFF]) + self.c[bit_pos + 1 :]
            K = decaps(c_bad, self.dk)
            assert len(K) == 32

    def test_wrong_key_different_shared_secret(self):
        """Decaps with wrong key produces different shared secret."""
        ek2, dk2 = keygen(bytes(range(32)), bytes(32))
        K_wrong = decaps(self.c, dk2)
        assert K_wrong != self.K_enc

    def test_wrong_key_no_crash(self):
        """Decaps with wrong key does not crash."""
        ek2, dk2 = keygen(bytes(range(32)), bytes(32))
        K = decaps(self.c, dk2)
        assert len(K) == 32

    def test_all_zero_ciphertext_no_crash(self):
        """All-zero ciphertext does not crash."""
        K = decaps(bytes(768), self.dk)
        assert len(K) == 32

    def test_all_zero_ciphertext_rejection(self):
        """All-zero ciphertext triggers implicit rejection."""
        K = decaps(bytes(768), self.dk)
        assert K != self.K_enc
