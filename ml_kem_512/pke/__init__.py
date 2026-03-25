"""
K-PKE (public-key encryption scheme) for ML-KEM-512.
"""

from ml_kem_512.pke.keygen import (
    compute_public_key_ntt,
    decode_public_key,
    decode_secret_key,
    encode_public_key,
    encode_secret_key,
    keygen_pke,
    sample_error_vector,
    sample_secret_vector,
)
from ml_kem_512.pke.matrix import generate_matrix

__all__ = [
    "generate_matrix",
    "sample_secret_vector",
    "sample_error_vector",
    "compute_public_key_ntt",
    "encode_public_key",
    "encode_secret_key",
    "decode_public_key",
    "decode_secret_key",
    "keygen_pke",
]
