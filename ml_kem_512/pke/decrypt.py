"""
K-PKE Decryption for ML-KEM-512.

Implements FIPS 203 Section 5.3 K-PKE.Decrypt:

  K-PKE.Decrypt(dk, c):
    Step 1: Decode secret key and ciphertext
      s_hat = ByteDecode_12(dk)
      u = Decompress_du(ByteDecode_du(c1))
      v = Decompress_dv(ByteDecode_dv(c2))

    Step 2: Recover message
      u_hat = NTT(u)
      w = v - InvNTT(s_hat^T . u_hat)
      m = ByteEncode_1(Compress_1(w))

    return m  (32 bytes)
"""

from __future__ import annotations

from ml_kem_512.encoding.codec import byte_encode
from ml_kem_512.encoding.compress import compress_poly
from ml_kem_512.module.vector import dot_ntt, ntt_vec
from ml_kem_512.pke.encrypt import decode_ciphertext
from ml_kem_512.pke.keygen import decode_secret_key
from ml_kem_512.polynomial.ntt import inv_ntt
from ml_kem_512.polynomial.poly import Polynomial


def decrypt_pke(dk: bytes, c: bytes) -> bytes:
    """
    K-PKE.Decrypt(dk, c) — FIPS 203 Section 5.3.

    Args:
        dk: 768-byte encoded secret key
        c:  768-byte ciphertext

    Returns:
        m: 32-byte recovered message
    """
    # Step 1: decode secret key and ciphertext
    s_hat = decode_secret_key(dk)
    u, v = decode_ciphertext(c)

    # Step 2: u_hat = NTT(u)
    u_hat = ntt_vec(u)

    # Step 3: w = v - InvNTT(s_hat^T . u_hat)
    inner = dot_ntt(s_hat, u_hat)
    inner_poly = inv_ntt(inner.coeffs)
    w = v - inner_poly

    # Step 4: m = ByteEncode_1(Compress_1(w))
    compressed = compress_poly(1, w)
    m = byte_encode(1, compressed)

    return m
