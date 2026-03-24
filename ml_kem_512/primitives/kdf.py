"""
Key Derivation Functions (KDF) for ML-KEM-512, as defined in FIPS 203.

These are named wrappers over SHA3/SHAKE primitives used at specific
points in the ML-KEM protocol:

  H(x) = SHA3-256(x)   -- hash public key ek before use
  G(x) = SHA3-512(x)   -- split into (rho, sigma) in KeyGen
                           and (K_bar, r) in Encaps
  J(x) = SHAKE-256(x)  -- derive final 32-byte shared secret K
"""

from __future__ import annotations

import hashlib


def H(data: bytes) -> bytes:
    """
    H(x) = SHA3-256(x)  (FIPS 203, Section 4.1)

    Used to hash the encapsulation key (public key) before it is
    concatenated with the random message m in Encaps:
        (K_bar, r) = G(m || H(ek))

    Args:
        data: arbitrary-length input bytes

    Returns:
        32 bytes (256-bit digest)
    """
    return hashlib.sha3_256(data).digest()


def G(data: bytes) -> tuple[bytes, bytes]:
    """
    G(x) = SHA3-512(x)  (FIPS 203, Section 4.1)

    Produces 64 bytes which are split into two 32-byte halves:
      - KeyGen:  (rho, sigma) = G(d || k)
      - Encaps:  (K_bar, r)   = G(m || H(ek))

    Args:
        data: arbitrary-length input bytes

    Returns:
        (left, right): two 32-byte halves of the SHA3-512 digest
    """
    digest = hashlib.sha3_512(data).digest()
    return digest[:32], digest[32:]


def J(data: bytes) -> bytes:
    """
    J(x) = SHAKE-256(x, 32)  (FIPS 203, Section 4.1)

    Used to derive the final 32-byte shared secret K:
      - On success:  K = J(K_bar || H(c))
      - On failure:  K = J(z    || H(c))   (implicit rejection)

    Args:
        data: arbitrary-length input bytes

    Returns:
        32 bytes shared secret
    """
    shake = hashlib.shake_256()
    shake.update(data)
    return shake.digest(32)
