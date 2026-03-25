"""
Uniform sampling for ML-KEM-512.

Implements FIPS 203 Algorithm 6 - SampleNTT.

SampleNTT(XOF(rho, i, j)):
  Reads bytes from SHAKE-128 stream seeded with (rho, i, j).
  Parses them as 12-bit values using rejection sampling.
  Accepts values in [0, q-1], rejects values >= q.
  Continues until 256 coefficients are collected.

The result is already in NTT domain — it IS the matrix element A[i][j].
"""

from __future__ import annotations

from ml_kem_512.polynomial.poly import N, Q
from ml_kem_512.primitives.prf import XOF


def sample_ntt(rho: bytes, i: int, j: int) -> list:
    """
    SampleNTT(rho, i, j) — FIPS 203 Algorithm 6.

    Generates a uniform random polynomial in NTT domain for matrix
    position A[i][j]. Uses rejection sampling over SHAKE-128 output.

    Parses bytes in pairs of 3 (24 bits) -> two 12-bit values:
      d1 = b[0] + 256 * (b[1] mod 16)
      d2 = (b[1] >> 4) + 16 * b[2]

    Accepts d if d < q, rejects otherwise.

    Args:
        rho: 32-byte public seed
        i:   row index (0 to k-1)
        j:   column index (0 to k-1)

    Returns:
        list of N=256 integers in [0, Q-1] — the NTT-domain polynomial
    """
    # Read bytes in chunks; 3 bytes -> up to 2 candidates
    # Worst-case acceptance rate: q/2^12 = 3329/4096 ~ 81%
    # So ~315 bytes needed on average; we read in 168-byte blocks (XOF rate)
    xof = XOF(rho, i, j)
    coeffs = []
    # Read enough bytes upfront; 504 bytes gives ~336 candidates, well above 256
    buf = xof.read(504)
    pos = 0

    while len(coeffs) < N:
        if pos + 2 >= len(buf):
            # Extend buffer if somehow not enough (very rare)
            buf = buf + xof.read(168)

        b0 = buf[pos]
        b1 = buf[pos + 1]
        b2 = buf[pos + 2]
        pos += 3

        d1 = b0 + 256 * (b1 % 16)
        d2 = (b1 >> 4) + 16 * b2

        if d1 < Q:
            coeffs.append(d1)
        if d2 < Q and len(coeffs) < N:
            coeffs.append(d2)

    return coeffs[:N]
