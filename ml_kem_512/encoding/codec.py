"""
Byte encoding and decoding for ML-KEM-512.

Implements FIPS 203 Algorithms 4 and 5:

  ByteEncode_d(F): pack N d-bit integers into bytes
  ByteDecode_d(B): unpack bytes into N d-bit integers

Bit packing is little-endian within each integer:
  - LSB of each value goes into the lowest available bit position
  - Values are packed consecutively with no gaps

Byte lengths produced:
  d=1  ->  32 bytes   (message encoding)
  d=4  ->  128 bytes  (v ciphertext component)
  d=10 ->  320 bytes  (u ciphertext component, per polynomial)
  d=12 ->  384 bytes  (public key t, per polynomial)

ML-KEM-512 key/ciphertext sizes:
  Public key  ek = ByteEncode_12(t_hat) || rho  = 384*2 + 32 = 800 bytes
  Secret key  dk = ByteEncode_12(s_hat)          = 384*2     = 768 bytes
  Ciphertext  c  = ByteEncode_10(u) || ByteEncode_4(v)
                 = 320*2 + 128                   = 768 bytes
"""

from __future__ import annotations

from ml_kem_512.polynomial.poly import N, Polynomial


def byte_encode(d: int, values: list) -> bytes:
    """
    ByteEncode_d(F) — FIPS 203 Algorithm 4.

    Packs N d-bit integers into a byte string of length 32*d bytes.

    Args:
        d: bits per value (1 <= d <= 12)
        values: list of N=256 integers, each in [0, 2^d - 1]

    Returns:
        bytes of length N*d//8 = 32*d bytes
    """
    if len(values) != N:
        raise ValueError(f"byte_encode expects {N} values, got {len(values)}")

    # Collect all bits (LSB first for each value)
    bits = []
    for v in values:
        for bit in range(d):
            bits.append((v >> bit) & 1)

    # Pack bits into bytes (8 bits per byte, LSB first)
    out = bytearray(N * d // 8)
    for i, bit in enumerate(bits):
        out[i // 8] |= bit << (i % 8)
    return bytes(out)


def byte_decode(d: int, data: bytes) -> list:
    """
    ByteDecode_d(B) — FIPS 203 Algorithm 5.

    Unpacks a byte string into N d-bit integers.

    Args:
        d: bits per value (1 <= d <= 12)
        data: bytes of length 32*d

    Returns:
        list of N=256 integers in [0, 2^d - 1]
    """
    expected = N * d // 8
    if len(data) != expected:
        raise ValueError(f"byte_decode expects {expected} bytes for d={d}, got {len(data)}")

    # Extract all bits (LSB first within each byte)
    bits = []
    for byte in data:
        for bit in range(8):
            bits.append((byte >> bit) & 1)

    # Reconstruct d-bit values
    values = []
    for i in range(N):
        v = 0
        for bit in range(d):
            v |= bits[i * d + bit] << bit
        values.append(v)
    return values


def encode_poly(d: int, poly: Polynomial) -> bytes:
    """
    Encode a polynomial's coefficients as d-bit packed bytes.

    Args:
        d: bits per coefficient
        poly: Polynomial with N coefficients in [0, 2^d - 1]

    Returns:
        32*d bytes
    """
    return byte_encode(d, poly.coeffs)


def decode_poly(d: int, data: bytes) -> Polynomial:
    """
    Decode d-bit packed bytes into a polynomial.

    Args:
        d: bits per coefficient
        data: 32*d bytes

    Returns:
        Polynomial with N coefficients in [0, 2^d - 1]
    """
    return Polynomial(byte_decode(d, data))
