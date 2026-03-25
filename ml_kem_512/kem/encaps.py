"""
ML-KEM.Encaps for ML-KEM-512.

Implements FIPS 203 Section 6.2:

  ML-KEM.Encaps(ek):
    m <- random 32 bytes
    (K_bar, r) = G(m || H(ek))
    c = K-PKE.Encrypt(ek, m, r)
    K = J(K_bar || H(c))
    return (K, c)
"""

from __future__ import annotations

import os

from ml_kem_512.pke.encrypt import encrypt_pke
from ml_kem_512.primitives.kdf import G, H, J


def encaps(ek: bytes, m: bytes = b"") -> tuple:
    """
    ML-KEM.Encaps(ek) — FIPS 203 Section 6.2.

    Args:
        ek: 800-byte encapsulation key (public key)
        m:  32-byte random message (generated randomly if empty)

    Returns:
        (K, c):
          K: 32-byte shared secret
          c: 768-byte ciphertext
    """
    if len(ek) != 800:
        raise ValueError(f"Encapsulation key must be 800 bytes, got {len(ek)}")

    if not m:
        m = os.urandom(32)

    if len(m) != 32:
        raise ValueError(f"Message m must be 32 bytes, got {len(m)}")

    # Derive randomness from m and public key hash
    k_bar, r = G(m + H(ek))

    # Encrypt
    c = encrypt_pke(ek, m, r)

    # Derive shared secret
    K = J(k_bar + H(c))

    return K, c
