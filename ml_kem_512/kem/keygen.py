"""
ML-KEM.KeyGen for ML-KEM-512.

Implements FIPS 203 Section 6.1:

  ML-KEM.KeyGen():
    d <- random 32 bytes
    z <- random 32 bytes
    (ek_pke, dk_pke) = K-PKE.KeyGen(d)
    ek = ek_pke                              (800 bytes)
    dk = dk_pke || ek || H(ek) || z         (1632 bytes)
    return (ek, dk)
"""

from __future__ import annotations

import os

from ml_kem_512.pke.keygen import keygen_pke
from ml_kem_512.primitives.kdf import H


def keygen(d: bytes = b"", z: bytes = b"") -> tuple:
    """
    ML-KEM.KeyGen() — FIPS 203 Section 6.1.

    Args:
        d: 32-byte seed (generated randomly if empty)
        z: 32-byte rejection seed (generated randomly if empty)

    Returns:
        (ek, dk):
          ek: 800-byte encapsulation key (public key)
          dk: 1632-byte decapsulation key (private key)
    """
    if not d:
        d = os.urandom(32)
    if not z:
        z = os.urandom(32)

    if len(d) != 32:
        raise ValueError(f"Seed d must be 32 bytes, got {len(d)}")
    if len(z) != 32:
        raise ValueError(f"Seed z must be 32 bytes, got {len(z)}")

    # K-PKE key generation
    ek_pke, dk_pke = keygen_pke(d)

    # ML-KEM decapsulation key: dk_pke || ek || H(ek) || z
    ek = ek_pke
    dk = dk_pke + ek + H(ek) + z

    return ek, dk
