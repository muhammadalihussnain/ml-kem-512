"""
ML-KEM-512 Key Encapsulation Mechanism.

Implements FIPS 203 ML-KEM.Encaps and ML-KEM.KeyGen wrapper.
"""

from ml_kem_512.kem.encaps import encaps
from ml_kem_512.kem.keygen import keygen

__all__ = ["keygen", "encaps"]
