"""
Centered Binomial Distribution (CBD) sampling for ML-KEM-512.

Implements FIPS 203 Algorithm 7 - SamplePolyCBD_eta.

CBD_eta(b):
  For each of the 256 coefficients:
    a = sum of eta bits  (from consecutive bits of b)
    b = sum of eta bits
    coefficient = a - b  (in range [-eta, eta])

ML-KEM-512 uses:
  eta1 = 3  -- secrets s, e in KeyGen
  eta2 = 2  -- randomness r, errors e1, e2 in Encaps

Input byte lengths:
  eta=3 -> 64*3 = 192 bytes  (PRF output length)
  eta=2 -> 64*2 = 128 bytes
"""

from __future__ import annotations

from ml_kem_512.polynomial.poly import N, Polynomial, Q

# ML-KEM-512 CBD parameters
ETA1 = 3  # for secret/error vectors in KeyGen
ETA2 = 2  # for randomness/errors in Encaps


def cbd(eta: int, data: bytes) -> Polynomial:
    """
    SamplePolyCBD_eta (FIPS 203 Algorithm 7).

    Samples a polynomial from the centered binomial distribution
    using the provided pseudorandom bytes.

    Each coefficient c_i = a_i - b_i where:
      a_i = sum of eta consecutive bits starting at bit 2*i*eta
      b_i = sum of eta consecutive bits starting at bit (2*i+1)*eta

    Args:
        eta: distribution parameter (2 or 3 for ML-KEM-512)
        data: pseudorandom bytes, must be exactly 64*eta bytes

    Returns:
        Polynomial with 256 coefficients in [0, Q-1]
        (internally a - b is reduced mod Q, so -eta..eta maps to [Q-eta, Q-1] U [0, eta])
    """
    expected = 64 * eta
    if len(data) != expected:
        raise ValueError(
            f"CBD requires exactly 64*eta={expected} bytes for eta={eta}, got {len(data)}"
        )
    if eta not in (2, 3):
        raise ValueError(f"ML-KEM-512 only uses eta=2 or eta=3, got {eta}")

    # Convert bytes to a flat bit array
    bits = _bytes_to_bits(data)

    coeffs = []
    for i in range(N):
        # a = sum of eta bits at positions [2*i*eta .. 2*i*eta + eta - 1]
        a = sum(bits[2 * i * eta + j] for j in range(eta))
        # b = sum of eta bits at positions [2*i*eta + eta .. 2*i*eta + 2*eta - 1]
        b = sum(bits[2 * i * eta + eta + j] for j in range(eta))
        # coefficient = a - b, reduced mod Q into [0, Q-1]
        coeffs.append((a - b) % Q)

    return Polynomial(coeffs)


def cbd_eta1(data: bytes) -> Polynomial:
    """
    CBD with eta1=3 — used for secret and error vectors in KeyGen.

    Args:
        data: 192 bytes from PRF(sigma, N)

    Returns:
        Polynomial sampled from CBD_3
    """
    return cbd(ETA1, data)


def cbd_eta2(data: bytes) -> Polynomial:
    """
    CBD with eta2=2 — used for randomness and errors in Encaps.

    Args:
        data: 128 bytes from PRF(r, N)

    Returns:
        Polynomial sampled from CBD_2
    """
    return cbd(ETA2, data)


def _bytes_to_bits(data: bytes) -> list:
    """
    Convert bytes to a list of bits (LSB first within each byte).

    FIPS 203 uses little-endian bit ordering within each byte.

    Args:
        data: input bytes

    Returns:
        list of 0/1 integers, length = 8 * len(data)
    """
    bits = []
    for byte in data:
        for bit_pos in range(8):
            bits.append((byte >> bit_pos) & 1)
    return bits
