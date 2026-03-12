"""
Cryptographic hash functions for ML-KEM-512.

This module provides wrappers around Python's hashlib implementation
of FIPS 202 (SHA-3 and SHAKE) functions used in ML-KEM.
"""

import hashlib


def shake128(data: bytes, output_length: int) -> bytes:
    """
    SHAKE-128 extendable output function.

    Args:
        data: Input bytes to hash
        output_length: Desired output length in bytes

    Returns:
        output_length bytes of pseudorandom output
    """
    shake = hashlib.shake_128()
    shake.update(data)
    return shake.digest(output_length)


def shake256(data: bytes, output_length: int) -> bytes:
    """
    SHAKE-256 extendable output function.

    Args:
        data: Input bytes to hash
        output_length: Desired output length in bytes

    Returns:
        output_length bytes of pseudorandom output
    """
    shake = hashlib.shake_256()
    shake.update(data)
    return shake.digest(output_length)


def sha3_256(data: bytes) -> bytes:
    """
    SHA3-256 hash function.

    Args:
        data: Input bytes to hash

    Returns:
        32 bytes (256 bits) hash output
    """
    return hashlib.sha3_256(data).digest()


def sha3_512(data: bytes) -> bytes:
    """
    SHA3-512 hash function.

    Args:
        data: Input bytes to hash

    Returns:
        64 bytes (512 bits) hash output
    """
    return hashlib.sha3_512(data).digest()
