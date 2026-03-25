"""
K-PKE Encryption for ML-KEM-512.

Implements FIPS 203 Section 5.2 K-PKE.Encrypt:

  K-PKE.Encrypt(ek, m, r):
    Step 1: Decode public key -> (t_hat, rho)
    Step 2: Regenerate matrix A^T from rho
    Step 3: Sample r_vec[i] = CBD_eta1(PRF(r, i)),  r_hat = NTT(r_vec)
    Step 4: Sample e1[i]    = CBD_eta2(PRF(r, k+i))
            Sample e2       = CBD_eta2(PRF(r, 2k))
    Step 5: u = InvNTT(A^T * r_hat) + e1
            mu = Decompress_1(ByteDecode_1(m))
            v  = InvNTT(t_hat^T . r_hat) + e2 + mu
    Step 6: c1 = ByteEncode_du(Compress_du(u))
            c2 = ByteEncode_dv(Compress_dv(v))
            c  = c1 || c2   (768 bytes)
"""

from __future__ import annotations

from ml_kem_512.encoding.codec import byte_decode, byte_encode
from ml_kem_512.encoding.compress import DU, DV, compress_poly, decompress
from ml_kem_512.module.vector import K, PolyVec, dot_ntt, inv_ntt_vec, mat_vec_mul, ntt_vec
from ml_kem_512.pke.keygen import decode_public_key
from ml_kem_512.pke.matrix import generate_matrix
from ml_kem_512.polynomial.ntt import inv_ntt
from ml_kem_512.polynomial.poly import N, Polynomial
from ml_kem_512.primitives.prf import prf
from ml_kem_512.sampling.cbd import ETA1, ETA2, cbd_eta1, cbd_eta2


def encrypt_pke(ek: bytes, m: bytes, r: bytes) -> bytes:
    """
    K-PKE.Encrypt(ek, m, r) — FIPS 203 Section 5.2.

    Args:
        ek: 800-byte encoded public key
        m:  32-byte message (plaintext)
        r:  32-byte encryption randomness seed

    Returns:
        768-byte ciphertext c = c1 || c2
    """
    if len(m) != 32:
        raise ValueError(f"Message m must be 32 bytes, got {len(m)}")
    if len(r) != 32:
        raise ValueError(f"Randomness r must be 32 bytes, got {len(r)}")

    # Step 1: decode public key
    t_hat, rho = decode_public_key(ek)

    # Step 2: regenerate A^T from rho
    At_hat = generate_matrix(rho, transpose=True)

    # Step 3: sample r_vec from CBD_eta1, transform to NTT
    r_polys = []
    for i in range(K):
        r_polys.append(cbd_eta1(prf(r, i, 64 * ETA1)))
    r_vec = PolyVec(r_polys)
    r_hat = ntt_vec(r_vec)

    # Step 4: sample error vectors e1 (vector) and e2 (scalar)
    e1_polys = []
    for i in range(K):
        e1_polys.append(cbd_eta2(prf(r, K + i, 64 * ETA2)))
    e1 = PolyVec(e1_polys)
    e2 = cbd_eta2(prf(r, 2 * K, 64 * ETA2))

    # Step 5a: u = InvNTT(A^T * r_hat) + e1
    u_hat = mat_vec_mul(At_hat, r_hat)
    u = inv_ntt_vec(u_hat) + e1

    # Step 5b: mu = Decompress_1(ByteDecode_1(m))
    m_bits = byte_decode(1, m)
    mu_coeffs = [decompress(1, b) for b in m_bits]
    mu = Polynomial(mu_coeffs)

    # Step 5c: v = InvNTT(t_hat^T . r_hat) + e2 + mu
    inner = dot_ntt(t_hat, r_hat)
    v_ntt_poly = inv_ntt(inner.coeffs)
    v = v_ntt_poly + e2 + mu

    # Step 6: compress and encode
    c1 = _encode_u(u)
    c2 = _encode_v(v)
    return c1 + c2


def _encode_u(u: PolyVec) -> bytes:
    """Compress and encode u vector: k * 320 = 640 bytes."""
    result = b""
    for i in range(K):
        compressed = compress_poly(DU, u[i])
        result += byte_encode(DU, compressed)
    return result


def _encode_v(v: Polynomial) -> bytes:
    """Compress and encode v scalar: 128 bytes."""
    compressed = compress_poly(DV, v)
    return byte_encode(DV, compressed)


def decode_ciphertext(c: bytes) -> tuple:
    """
    Decode ciphertext c into (u, v).

    Args:
        c: 768-byte ciphertext

    Returns:
        (u, v): PolyVec and Polynomial in decompressed form
    """
    if len(c) != 768:
        raise ValueError(f"Ciphertext must be 768 bytes, got {len(c)}")

    from ml_kem_512.encoding.compress import decompress_poly

    # c1 = first 640 bytes (u vector, k * 320)
    c1 = c[:640]
    # c2 = last 128 bytes (v scalar)
    c2 = c[640:]

    u_polys = []
    for i in range(K):
        chunk = c1[i * 320 : (i + 1) * 320]
        values = byte_decode(DU, chunk)
        u_polys.append(decompress_poly(DU, values))
    u = PolyVec(u_polys)

    v_values = byte_decode(DV, c2)
    v = decompress_poly(DV, v_values)

    return u, v
