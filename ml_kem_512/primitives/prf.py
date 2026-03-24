"""
Pseudorandom Functions (PRF) and Extendable Output Functions (XOF)
for ML-KEM-512, as defined in FIPS 203.

PRF: SHAKE-256 based pseudorandom function used for CBD sampling.
XOF: SHAKE-128 based extendable output function used for matrix generation.
"""

import hashlib


def prf(s: bytes, b: int, output_length: int) -> bytes:
    """
    PRF(s, b) — Pseudorandom Function (FIPS 203, Section 4.1).

    Defined as: PRF_η(s, b) = SHAKE-256(s || b, 64*η)
    Used to generate pseudorandom bytes for CBD sampling.

    Args:
        s: 32-byte seed (sigma σ from KeyGen)
        b: 1-byte counter (domain separation)
        output_length: number of output bytes to produce

    Returns:
        output_length pseudorandom bytes
    """
    if len(s) != 32:
        raise ValueError(f"PRF seed must be 32 bytes, got {len(s)}")
    if b < 0 or b > 255:
        raise ValueError(f"PRF counter b must be in [0, 255], got {b}")

    shake = hashlib.shake_256()
    shake.update(s)
    shake.update(bytes([b]))
    return shake.digest(output_length)


class XOF:
    """
    XOF(ρ, i, j) — Extendable Output Function (FIPS 203, Section 4.2.2).

    Defined as: XOF(ρ, i, j) = SHAKE-128(ρ || j || i)
    Used to generate matrix elements A[i][j] via SampleNTT.

    Note: FIPS 203 uses the order ρ || i || j in the spec text,
    but the actual byte encoding is ρ || j || i for A[i][j].
    We follow the spec: XOF(ρ, i, j) seeds with ρ || i || j.
    """

    def __init__(self, rho: bytes, i: int, j: int):
        """
        Initialize XOF stream for matrix position (i, j).

        Args:
            rho: 32-byte public seed ρ
            i: row index (0 to k-1)
            j: column index (0 to k-1)
        """
        if len(rho) != 32:
            raise ValueError(f"XOF seed rho must be 32 bytes, got {len(rho)}")
        if i < 0 or i > 255:
            raise ValueError(f"XOF index i must be in [0, 255], got {i}")
        if j < 0 or j > 255:
            raise ValueError(f"XOF index j must be in [0, 255], got {j}")

        self._rho = rho
        self._shake = hashlib.shake_128()
        self._shake.update(rho)
        self._shake.update(bytes([i]))
        self._shake.update(bytes([j]))

    def read(self, n: int) -> bytes:
        """
        Read n bytes from the XOF stream.

        Args:
            n: number of bytes to read

        Returns:
            n bytes of pseudorandom output
        """
        return self._shake.digest(n)

    def seed(self) -> bytes:
        """
        Return the 32-byte seed used to initialize this XOF stream.

        Returns:
            The rho seed bytes
        """
        return self._rho


def xof(rho: bytes, i: int, j: int, output_length: int) -> bytes:
    """
    Convenience wrapper: XOF(ρ, i, j) → output_length bytes.

    Args:
        rho: 32-byte public seed ρ
        i: row index
        j: column index
        output_length: number of bytes to generate

    Returns:
        output_length bytes from SHAKE-128(ρ || i || j)
    """
    return XOF(rho, i, j).read(output_length)
