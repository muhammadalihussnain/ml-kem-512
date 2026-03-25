"""
K-PKE Key Generation components for ML-KEM-512.

Implements FIPS 203 Section 5.1 K-PKE.KeyGen steps:

  Step 3-4: Secret vector s
    s[i] = CBD_eta1(PRF(sigma, i))   for i in 0..k-1
    s_hat = NTT(s)

  Step 5-6: Error vector e
    e[i] = CBD_eta1(PRF(sigma, k+i)) for i in 0..k-1
    e_hat = NTT(e)

  Step 7: Public key t_hat
    t_hat = A_hat * s_hat + e_hat

  Step 8: Encode keys
    pk = ByteEncode_12(t_hat) || rho   (800 bytes)
    sk = ByteEncode_12(s_hat)          (768 bytes)
"""

from __future__ import annotations

from ml_kem_512.encoding.codec import byte_decode, byte_encode
from ml_kem_512.module.vector import K, PolyVec, mat_vec_mul, ntt_vec
from ml_kem_512.polynomial.poly import Polynomial
from ml_kem_512.primitives.prf import prf
from ml_kem_512.sampling.cbd import ETA1, cbd_eta1


def sample_secret_vector(sigma: bytes, counter_start: int = 0) -> PolyVec:
    """
    Sample secret vector s from CBD_eta1 (FIPS 203 KeyGen Steps 3-4).

    s[i] = CBD_eta1(PRF(sigma, counter_start + i))  for i in 0..k-1

    Args:
        sigma: 32-byte seed (second half of G(d))
        counter_start: starting counter value (0 for s, k for e)

    Returns:
        PolyVec of k polynomials in polynomial domain (NOT NTT)
    """
    if len(sigma) != 32:
        raise ValueError(f"sigma must be 32 bytes, got {len(sigma)}")

    polys = []
    for i in range(K):
        prf_out = prf(sigma, counter_start + i, 64 * ETA1)
        polys.append(cbd_eta1(prf_out))
    return PolyVec(polys)


def sample_error_vector(sigma: bytes) -> PolyVec:
    """
    Sample error vector e from CBD_eta1 (FIPS 203 KeyGen Steps 5-6).

    e[i] = CBD_eta1(PRF(sigma, k + i))  for i in 0..k-1

    Args:
        sigma: 32-byte seed

    Returns:
        PolyVec of k polynomials in polynomial domain (NOT NTT)
    """
    return sample_secret_vector(sigma, counter_start=K)


def compute_public_key_ntt(A_hat: list, s_hat: PolyVec, e_hat: PolyVec) -> PolyVec:
    """
    Compute t_hat = A_hat * s_hat + e_hat (FIPS 203 KeyGen Step 7).

    All inputs and output are in NTT domain.

    Args:
        A_hat: k x k matrix of NTT coefficient lists
        s_hat: secret vector in NTT domain
        e_hat: error vector in NTT domain

    Returns:
        t_hat = A*s + e in NTT domain
    """
    As_hat = mat_vec_mul(A_hat, s_hat)
    return As_hat + e_hat


def encode_public_key(t_hat: PolyVec, rho: bytes) -> bytes:
    """
    Encode public key: ek = ByteEncode_12(t_hat) || rho (800 bytes).

    Args:
        t_hat: public key vector in NTT domain (k polynomials)
        rho: 32-byte public seed

    Returns:
        800-byte encoded public key
    """
    encoded = b"".join(byte_encode(12, t_hat[i].coeffs) for i in range(K))
    return encoded + rho


def encode_secret_key(s_hat: PolyVec) -> bytes:
    """
    Encode secret key: dk = ByteEncode_12(s_hat) (768 bytes).

    Args:
        s_hat: secret vector in NTT domain (k polynomials)

    Returns:
        768-byte encoded secret key
    """
    return b"".join(byte_encode(12, s_hat[i].coeffs) for i in range(K))


def decode_public_key(ek: bytes) -> tuple:
    """
    Decode public key bytes back into (t_hat, rho).

    Args:
        ek: 800-byte encoded public key

    Returns:
        (t_hat, rho): PolyVec in NTT domain and 32-byte seed
    """
    if len(ek) != 800:
        raise ValueError(f"Public key must be 800 bytes, got {len(ek)}")
    polys = []
    for i in range(K):
        chunk = ek[i * 384 : (i + 1) * 384]
        polys.append(Polynomial(byte_decode(12, chunk)))
    t_hat = PolyVec(polys)
    rho = ek[768:]
    return t_hat, rho


def decode_secret_key(dk: bytes) -> PolyVec:
    """
    Decode secret key bytes back into s_hat.

    Args:
        dk: 768-byte encoded secret key

    Returns:
        s_hat: PolyVec in NTT domain
    """
    if len(dk) != 768:
        raise ValueError(f"Secret key must be 768 bytes, got {len(dk)}")
    polys = []
    for i in range(K):
        chunk = dk[i * 384 : (i + 1) * 384]
        polys.append(Polynomial(byte_decode(12, chunk)))
    return PolyVec(polys)
