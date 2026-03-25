"""
Compression and decompression for ML-KEM-512.

Implements FIPS 203 Algorithms 4 and 5:

  Compress_d(x) = round(2^d / q * x) mod 2^d
  Decompress_d(y) = round(q / 2^d * y)

where round(v) = floor(v + 0.5) = (2v + 1) // 2  (integer rounding).

ML-KEM-512 parameters:
  d_u = 10  -- compression bits for ciphertext vector u
  d_v =  4  -- compression bits for ciphertext scalar v
  d_t = 12  -- no compression (full coefficients for public key t)

These functions operate on individual coefficients (integers).
Polynomial-level wrappers compress/decompress all 256 coefficients.
"""

from __future__ import annotations

from ml_kem_512.polynomial.poly import N, Polynomial, Q

# ML-KEM-512 compression parameters
DU = 10  # bits for u vector
DV = 4  # bits for v scalar


def compress(d: int, x: int) -> int:
    """
    Compress_d(x) — FIPS 203 Algorithm 4 (single coefficient).

    Maps x in [0, Q-1] to a d-bit integer in [0, 2^d - 1].

    Formula: round(2^d / q * x) mod 2^d
    Integer form: ((x << d) + Q // 2) // Q  mod 2^d

    Args:
        d: number of output bits (1 <= d <= 12)
        x: input coefficient in [0, Q-1]

    Returns:
        integer in [0, 2^d - 1]
    """
    # round(2^d * x / q) mod 2^d
    # = ((2^d * x + q//2) // q) mod 2^d
    return ((x << d) + Q // 2) // Q % (1 << d)


def decompress(d: int, y: int) -> int:
    """
    Decompress_d(y) — FIPS 203 Algorithm 5 (single coefficient).

    Maps a d-bit integer y back to an approximate value in [0, Q-1].

    Formula: round(q / 2^d * y)
    Integer form: (q * y + 2^(d-1)) >> d

    Args:
        d: number of input bits (1 <= d <= 12)
        y: compressed value in [0, 2^d - 1]

    Returns:
        integer in [0, Q-1]
    """
    # round(q * y / 2^d) = (q * y + 2^(d-1)) // 2^d
    return (Q * y + (1 << (d - 1))) >> d


def compress_poly(d: int, poly: Polynomial) -> list:
    """
    Compress all 256 coefficients of a polynomial.

    Args:
        d: number of output bits per coefficient
        poly: input polynomial with coefficients in [0, Q-1]

    Returns:
        list of 256 integers in [0, 2^d - 1]
    """
    return [compress(d, c) for c in poly.coeffs]


def decompress_poly(d: int, values: list) -> Polynomial:
    """
    Decompress a list of d-bit values back into a polynomial.

    Args:
        d: number of bits per value
        values: list of 256 integers in [0, 2^d - 1]

    Returns:
        Polynomial with coefficients in [0, Q-1]
    """
    return Polynomial([decompress(d, v) for v in values])
