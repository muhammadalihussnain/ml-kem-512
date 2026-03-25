"""
ML-KEM.Decaps for ML-KEM-512.

Implements FIPS 203 Section 6.3:

  ML-KEM.Decaps(c, dk):
    Step 1: Parse dk = dk_pke(768) || ek(800) || h(32) || z(32)
    Step 2: m' = K-PKE.Decrypt(dk_pke, c)
    Step 3: (K_bar', r') = G(m' || h)
    Step 4: c' = K-PKE.Encrypt(ek, m', r')
    Step 5: if c == c':
              K = J(K_bar' || H(c))   # success
            else:
              K = J(z || H(c))        # implicit rejection
    return K  (32 bytes)
"""

from __future__ import annotations

from ml_kem_512.pke.decrypt import decrypt_pke
from ml_kem_512.pke.encrypt import encrypt_pke
from ml_kem_512.primitives.kdf import G, H, J

# dk layout (1632 bytes total)
_DK_PKE_LEN = 768
_EK_LEN = 800
_H_LEN = 32
_Z_LEN = 32


def _parse_dk(dk: bytes) -> tuple:
    """Parse 1632-byte decapsulation key into (dk_pke, ek, h, z)."""
    if len(dk) != 1632:
        raise ValueError(f"Decapsulation key must be 1632 bytes, got {len(dk)}")
    dk_pke = dk[:_DK_PKE_LEN]
    ek = dk[_DK_PKE_LEN : _DK_PKE_LEN + _EK_LEN]
    h = dk[_DK_PKE_LEN + _EK_LEN : _DK_PKE_LEN + _EK_LEN + _H_LEN]
    z = dk[_DK_PKE_LEN + _EK_LEN + _H_LEN :]
    return dk_pke, ek, h, z


def decaps(c: bytes, dk: bytes) -> bytes:
    """
    ML-KEM.Decaps(c, dk) — FIPS 203 Section 6.3.

    Args:
        c:  768-byte ciphertext
        dk: 1632-byte decapsulation key

    Returns:
        K: 32-byte shared secret
    """
    if len(c) != 768:
        raise ValueError(f"Ciphertext must be 768 bytes, got {len(c)}")

    # Step 1: parse decapsulation key
    dk_pke, ek, h, z = _parse_dk(dk)

    # Step 2: decrypt to recover candidate message
    m_prime = decrypt_pke(dk_pke, c)

    # Step 3: re-derive randomness
    k_bar_prime, r_prime = G(m_prime + h)

    # Step 4: re-encrypt to verify
    c_prime = encrypt_pke(ek, m_prime, r_prime)

    # Step 5: implicit rejection — constant-time comparison
    h_c = H(c)
    if _ct_eq(c, c_prime):
        return J(k_bar_prime + h_c)
    else:
        return J(z + h_c)


def _ct_eq(a: bytes, b: bytes) -> bool:
    """
    Constant-time equality check.
    Returns True if a == b, without short-circuiting.
    """
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= x ^ y
    return diff == 0
