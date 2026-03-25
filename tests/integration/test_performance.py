"""
Milestone 8.4: Performance Benchmarks for ML-KEM-512.

Measures and verifies:
  - KeyGen time   < 5000ms (generous for Python)
  - Encaps time   < 5000ms
  - Decaps time   < 5000ms
  - Key sizes match specification
  - Ciphertext size matches specification

These are timing sanity checks, not strict performance gates.
The target is to confirm the implementation is not pathologically slow.
"""

import time

from ml_kem_512.kem.decaps import decaps
from ml_kem_512.kem.encaps import encaps
from ml_kem_512.kem.keygen import keygen

D = bytes(range(32))
Z = bytes(range(1, 33))
MSG = bytes([0x42] * 32)

# Generous upper bound for pure-Python unoptimized implementation (ms)
MAX_MS = 5000


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------------------
# Size verification (from plan)
# ---------------------------------------------------------------------------


class TestSizes:
    def test_ek_size(self):
        """Encapsulation key is 800 bytes."""
        ek, _ = keygen(D, Z)
        assert len(ek) == 800

    def test_dk_size(self):
        """Decapsulation key is 1632 bytes."""
        _, dk = keygen(D, Z)
        assert len(dk) == 1632

    def test_ciphertext_size(self):
        """Ciphertext is 768 bytes."""
        ek, _ = keygen(D, Z)
        _, c = encaps(ek, MSG)
        assert len(c) == 768

    def test_shared_secret_size(self):
        """Shared secret is 32 bytes."""
        ek, dk = keygen(D, Z)
        K, c = encaps(ek, MSG)
        assert len(K) == 32


# ---------------------------------------------------------------------------
# Timing tests
# ---------------------------------------------------------------------------


class TestTiming:
    def test_keygen_under_limit(self):
        """KeyGen completes within time limit."""
        t = time.perf_counter()
        keygen(D, Z)
        elapsed = _ms(t)
        assert elapsed < MAX_MS, f"KeyGen took {elapsed:.1f}ms, limit={MAX_MS}ms"

    def test_encaps_under_limit(self):
        """Encaps completes within time limit."""
        ek, _ = keygen(D, Z)
        t = time.perf_counter()
        encaps(ek, MSG)
        elapsed = _ms(t)
        assert elapsed < MAX_MS, f"Encaps took {elapsed:.1f}ms, limit={MAX_MS}ms"

    def test_decaps_under_limit(self):
        """Decaps completes within time limit."""
        ek, dk = keygen(D, Z)
        _, c = encaps(ek, MSG)
        t = time.perf_counter()
        decaps(c, dk)
        elapsed = _ms(t)
        assert elapsed < MAX_MS, f"Decaps took {elapsed:.1f}ms, limit={MAX_MS}ms"

    def test_full_flow_under_limit(self):
        """Full KeyGen + Encaps + Decaps completes within 3x limit."""
        t = time.perf_counter()
        ek, dk = keygen(D, Z)
        K1, c = encaps(ek, MSG)
        K2 = decaps(c, dk)
        elapsed = _ms(t)
        assert K1 == K2
        assert elapsed < MAX_MS * 3, f"Full flow took {elapsed:.1f}ms"
