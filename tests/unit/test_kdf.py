"""
Unit tests for Key Derivation Functions (Milestone 1.3).

Tests cover:
  H(x) = SHA3-256(x)      -- public key hashing
  G(x) = SHA3-512(x)      -- seed expansion (KeyGen / Encaps)
  J(x) = SHAKE-256(x, 32) -- shared secret derivation

Verification strategy:
- Output lengths are correct
- Determinism: same input -> same output
- Cross-check against raw hashlib (ground-truth)
- G splits correctly into two 32-byte halves
- Protocol usage patterns: KeyGen seed split, Encaps flow, Decaps rejection
- Different inputs produce different outputs
- Arbitrary-length inputs are accepted
"""

import hashlib

from ml_kem_512.primitives.kdf import G, H, J


# ---------------------------------------------------------------------------
# Reference implementations (raw hashlib, no abstraction layer)
# ---------------------------------------------------------------------------


def _ref_H(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def _ref_G(data: bytes) -> tuple:
    d = hashlib.sha3_512(data).digest()
    return d[:32], d[32:]


def _ref_J(data: bytes) -> bytes:
    h = hashlib.shake_256()
    h.update(data)
    return h.digest(32)


# ---------------------------------------------------------------------------
# H tests
# ---------------------------------------------------------------------------


class TestH:
    """H(x) = SHA3-256(x) -- used to hash the encapsulation key."""

    def test_output_length(self):
        """H always returns exactly 32 bytes."""
        assert len(H(b"")) == 32
        assert len(H(b"abc")) == 32
        assert len(H(b"x" * 1000)) == 32

    def test_empty_input(self):
        """H matches SHA3-256 NIST vector for empty input."""
        expected = bytes.fromhex(
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        )
        assert H(b"") == expected

    def test_known_vector(self):
        """H matches SHA3-256 NIST vector for 'abc'."""
        expected = bytes.fromhex(
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        )
        assert H(b"abc") == expected

    def test_deterministic(self):
        """Same input always produces same output."""
        data = b"encapsulation key bytes"
        assert H(data) == H(data)

    def test_different_inputs_differ(self):
        """Different inputs produce different digests."""
        assert H(b"key1") != H(b"key2")

    def test_cross_check_raw_sha3_256(self):
        """H output matches raw hashlib SHA3-256."""
        for msg in [b"", b"test", b"x" * 800, bytes(range(64))]:
            assert H(msg) == _ref_H(msg)

    def test_encaps_usage_pattern(self):
        """
        Simulate Encaps usage: (K_bar, r) = G(m || H(ek))
        H(ek) must be 32 bytes so it can be concatenated with m.
        """
        ek = bytes(i % 256 for i in range(800))  # ML-KEM-512 public key is 800 bytes
        h_ek = H(ek)
        assert len(h_ek) == 32
        m = bytes(32)
        combined = m + h_ek
        assert len(combined) == 64


# ---------------------------------------------------------------------------
# G tests
# ---------------------------------------------------------------------------


class TestG:
    """G(x) = SHA3-512(x) split into two 32-byte halves."""

    def test_output_is_two_32_byte_values(self):
        """G returns a tuple of two 32-byte values."""
        left, right = G(b"seed")
        assert len(left) == 32
        assert len(right) == 32

    def test_empty_input(self):
        """G matches SHA3-512 NIST vector for empty input, split correctly."""
        full = bytes.fromhex(
            "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6"
            "15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
        )
        left, right = G(b"")
        assert left == full[:32]
        assert right == full[32:]

    def test_halves_are_different(self):
        """Left and right halves of G output are different."""
        left, right = G(b"test seed")
        assert left != right

    def test_deterministic(self):
        """Same input always produces same (left, right) pair."""
        data = b"deterministic seed"
        assert G(data) == G(data)

    def test_different_inputs_differ(self):
        """Different inputs produce different outputs."""
        l1, r1 = G(b"seed_a")
        l2, r2 = G(b"seed_b")
        assert l1 != l2
        assert r1 != r2

    def test_cross_check_raw_sha3_512(self):
        """G output matches raw hashlib SHA3-512 split at 32 bytes."""
        for msg in [b"", b"test", bytes(range(32)), b"x" * 100]:
            assert G(msg) == _ref_G(msg)

    def test_keygen_usage_pattern(self):
        """
        Simulate KeyGen: (rho, sigma) = G(d || k)
        d is 32 random bytes, k=2 for ML-KEM-512.
        rho seeds matrix A, sigma seeds secret/error vectors.
        """
        d = bytes(range(32))
        k_param = bytes([2])  # ML-KEM-512 parameter k=2
        rho, sigma = G(d + k_param)
        assert len(rho) == 32
        assert len(sigma) == 32
        assert rho != sigma

    def test_encaps_usage_pattern(self):
        """
        Simulate Encaps: (K_bar, r) = G(m || H(ek))
        K_bar feeds into final shared secret, r seeds encryption randomness.
        """
        m = bytes(32)
        h_ek = bytes(range(32))  # simulated H(ek)
        k_bar, r = G(m + h_ek)
        assert len(k_bar) == 32
        assert len(r) == 32

    def test_arbitrary_length_input(self):
        """G accepts inputs of any length."""
        for size in [0, 1, 32, 64, 100, 800]:
            left, right = G(bytes(size))
            assert len(left) == 32
            assert len(right) == 32


# ---------------------------------------------------------------------------
# J tests
# ---------------------------------------------------------------------------


class TestJ:
    """J(x) = SHAKE-256(x, 32) -- derives the final shared secret K."""

    def test_output_length(self):
        """J always returns exactly 32 bytes."""
        assert len(J(b"")) == 32
        assert len(J(b"abc")) == 32
        assert len(J(b"x" * 1000)) == 32

    def test_deterministic(self):
        """Same input always produces same output."""
        data = b"K_bar || H(c)"
        assert J(data) == J(data)

    def test_different_inputs_differ(self):
        """Different inputs produce different shared secrets."""
        assert J(b"input_a") != J(b"input_b")

    def test_cross_check_raw_shake256(self):
        """J output matches raw SHAKE-256(x, 32)."""
        for msg in [b"", b"test", bytes(range(64)), b"z" * 200]:
            assert J(msg) == _ref_J(msg)

    def test_j_differs_from_h(self):
        """J (SHAKE-256) and H (SHA3-256) produce different outputs."""
        data = b"same input"
        assert J(data) != H(data)

    def test_decaps_success_pattern(self):
        """
        Simulate Decaps success: K = J(K_bar || H(c))
        K_bar is 32 bytes from G, H(c) is 32 bytes hash of ciphertext.
        Final shared secret K is 32 bytes.
        """
        k_bar = bytes(range(32))
        h_c = bytes(range(32, 64))  # simulated H(ciphertext)
        K = J(k_bar + h_c)
        assert len(K) == 32

    def test_decaps_rejection_pattern(self):
        """
        Simulate Decaps implicit rejection: K = J(z || H(c))
        z is the 32-byte random rejection value stored in dk.
        Success and rejection paths must produce different K.
        """
        z = bytes(range(32))
        k_bar = bytes(range(32, 64))
        h_c = bytes(range(64, 96))

        K_success = J(k_bar + h_c)
        K_reject = J(z + h_c)

        assert len(K_success) == 32
        assert len(K_reject) == 32
        assert K_success != K_reject

    def test_arbitrary_length_input(self):
        """J accepts inputs of any length."""
        for size in [0, 1, 32, 64, 100]:
            assert len(J(bytes(size))) == 32


# ---------------------------------------------------------------------------
# Cross-function tests
# ---------------------------------------------------------------------------


class TestKDFCrossFunction:
    """Verify H, G, J are independent and correctly separated."""

    def test_all_three_differ_on_same_input(self):
        """H, G (left half), and J produce different outputs for same input."""
        data = b"same data"
        h_out = H(data)
        g_left, _ = G(data)
        j_out = J(data)
        assert h_out != g_left
        assert h_out != j_out
        assert g_left != j_out

    def test_full_keygen_to_encaps_flow(self):
        """
        Simulate the full KDF chain:
          KeyGen: (rho, sigma) = G(d || k)
          Encaps: h_ek = H(ek), (K_bar, r) = G(m || h_ek)
          Encaps: K = J(K_bar || H(c))
        All outputs have correct sizes.
        """
        # KeyGen seed expansion
        d = bytes(range(32))
        rho, sigma = G(d + bytes([2]))
        assert len(rho) == 32
        assert len(sigma) == 32

        # Encaps: hash the public key
        ek = bytes(800)  # 800-byte ML-KEM-512 public key
        h_ek = H(ek)
        assert len(h_ek) == 32

        # Encaps: derive randomness
        m = bytes(32)
        k_bar, r = G(m + h_ek)
        assert len(k_bar) == 32
        assert len(r) == 32

        # Derive shared secret
        c = bytes(768)  # 768-byte ML-KEM-512 ciphertext
        K = J(k_bar + H(c))
        assert len(K) == 32
