"""
Generic ML-KEM engine supporting ML-KEM-512, ML-KEM-768, ML-KEM-1024.

Parameters per FIPS 203:
  Variant   k   eta1  eta2  du  dv
  512       2    3     2    10   4
  768       3    2     2    10   4
  1024      4    2     2    11   5
"""

from __future__ import annotations

import hashlib
import os

# ---------------------------------------------------------------------------
# Parameter sets
# ---------------------------------------------------------------------------

PARAMS = {
    "ML-KEM-512": {"k": 2, "eta1": 3, "eta2": 2, "du": 10, "dv": 4, "q": 3329, "n": 256},
    "ML-KEM-768": {"k": 3, "eta1": 2, "eta2": 2, "du": 10, "dv": 4, "q": 3329, "n": 256},
    "ML-KEM-1024": {"k": 4, "eta1": 2, "eta2": 2, "du": 11, "dv": 5, "q": 3329, "n": 256},
}

Q = 3329
N = 256
ZETA = 17


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _sha3_512(data: bytes):
    d = hashlib.sha3_512(data).digest()
    return d[:32], d[32:]

def _sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()

def _shake256(data: bytes, length: int) -> bytes:
    h = hashlib.shake_256(); h.update(data); return h.digest(length)

def _shake128(data: bytes, length: int) -> bytes:
    h = hashlib.shake_128(); h.update(data); return h.digest(length)

def _prf(s: bytes, b: int, length: int) -> bytes:
    h = hashlib.shake_256(); h.update(s); h.update(bytes([b])); return h.digest(length)

def _G(data: bytes):
    return _sha3_512(data)

def _H(data: bytes) -> bytes:
    return _sha3_256(data)

def _J(data: bytes) -> bytes:
    return _shake256(data, 32)


# ---------------------------------------------------------------------------
# NTT
# ---------------------------------------------------------------------------

def _bit_rev_7(n: int) -> int:
    r = 0
    for _ in range(7):
        r = (r << 1) | (n & 1); n >>= 1
    return r

_ZETAS = [pow(ZETA, _bit_rev_7(k), Q) for k in range(128)]
_N_INV = pow(128, -1, Q)

def _ntt(f):
    f = list(f); k = 1; length = 128
    while length >= 2:
        for start in range(0, N, 2 * length):
            z = _ZETAS[k]; k += 1
            for j in range(start, start + length):
                t = z * f[j + length] % Q
                f[j + length] = (f[j] - t) % Q
                f[j] = (f[j] + t) % Q
        length //= 2
    return f

def _inv_ntt(f):
    f = list(f); k = 127; length = 2
    while length <= 128:
        for start in range(0, N, 2 * length):
            z = _ZETAS[k]; k -= 1
            for j in range(start, start + length):
                t = f[j]
                f[j] = (t + f[j + length]) % Q
                f[j + length] = z * (f[j + length] - t) % Q
        length *= 2
    return [c * _N_INV % Q for c in f]

def _ntt_mul(a, b):
    r = [0] * N
    for i in range(128):
        a0, a1 = a[2*i], a[2*i+1]; b0, b1 = b[2*i], b[2*i+1]
        g = pow(ZETA, 2 * _bit_rev_7(i) + 1, Q)
        r[2*i] = (a0*b0 + a1*b1*g) % Q
        r[2*i+1] = (a0*b1 + a1*b0) % Q
    return r

def _poly_add(a, b): return [(x + y) % Q for x, y in zip(a, b)]
def _poly_sub(a, b): return [(x - y) % Q for x, y in zip(a, b)]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _bytes_to_bits(data: bytes):
    bits = []
    for byte in data:
        for i in range(8): bits.append((byte >> i) & 1)
    return bits

def _cbd(eta: int, data: bytes) -> list:
    bits = _bytes_to_bits(data)
    coeffs = []
    for i in range(N):
        a = sum(bits[2*i*eta + j] for j in range(eta))
        b = sum(bits[2*i*eta + eta + j] for j in range(eta))
        coeffs.append((a - b) % Q)
    return coeffs

def _sample_ntt(rho: bytes, i: int, j: int) -> list:
    h = hashlib.shake_128(); h.update(rho); h.update(bytes([i])); h.update(bytes([j]))
    buf = h.digest(504); pos = 0; coeffs = []
    while len(coeffs) < N:
        if pos + 2 >= len(buf): buf += h.digest(168)
        b0, b1, b2 = buf[pos], buf[pos+1], buf[pos+2]; pos += 3
        d1 = b0 + 256 * (b1 % 16); d2 = (b1 >> 4) + 16 * b2
        if d1 < Q: coeffs.append(d1)
        if d2 < Q and len(coeffs) < N: coeffs.append(d2)
    return coeffs[:N]


# ---------------------------------------------------------------------------
# Compression / Encoding
# ---------------------------------------------------------------------------

def _compress(d: int, x: int) -> int:
    return ((x << d) + Q // 2) // Q % (1 << d)

def _decompress(d: int, y: int) -> int:
    return (Q * y + (1 << (d - 1))) >> d

def _byte_encode(d: int, values: list) -> bytes:
    bits = []
    for v in values:
        for bit in range(d): bits.append((v >> bit) & 1)
    out = bytearray(N * d // 8)
    for i, bit in enumerate(bits): out[i // 8] |= bit << (i % 8)
    return bytes(out)

def _byte_decode(d: int, data: bytes) -> list:
    bits = []
    for byte in data:
        for bit in range(8): bits.append((byte >> bit) & 1)
    values = []
    for i in range(N):
        v = 0
        for bit in range(d): v |= bits[i * d + bit] << bit
        values.append(v)
    return values


# ---------------------------------------------------------------------------
# Core ML-KEM operations (generic over k, eta1, eta2, du, dv)
# ---------------------------------------------------------------------------

class MLKEM:
    def __init__(self, variant: str):
        p = PARAMS[variant]
        self.variant = variant
        self.k = p["k"]
        self.eta1 = p["eta1"]
        self.eta2 = p["eta2"]
        self.du = p["du"]
        self.dv = p["dv"]

    # --- Key sizes ---
    @property
    def ek_size(self): return 384 * self.k + 32
    @property
    def dk_pke_size(self): return 384 * self.k
    @property
    def dk_size(self): return self.dk_pke_size + self.ek_size + 32 + 32
    @property
    def ct_size(self): return 32 * self.du * self.k + 32 * self.dv

    def _gen_matrix(self, rho: bytes, transpose: bool = False) -> list:
        return [[_sample_ntt(rho, j if transpose else i, i if transpose else j)
                 for j in range(self.k)] for i in range(self.k)]

    def _sample_vec(self, sigma: bytes, start: int, eta: int) -> list:
        return [_cbd(eta, _prf(sigma, start + i, 64 * eta)) for i in range(self.k)]

    def _mat_vec_ntt(self, A, v_hat):
        result = []
        for i in range(self.k):
            s = [0] * N
            for j in range(self.k):
                p = _ntt_mul(A[i][j], v_hat[j])
                s = _poly_add(s, p)
            result.append(s)
        return result

    def _dot_ntt(self, a_hat, b_hat):
        s = [0] * N
        for i in range(self.k):
            s = _poly_add(s, _ntt_mul(a_hat[i], b_hat[i]))
        return s

    def keygen_pke(self, d: bytes):
        rho, sigma = _G(d + bytes([self.k]))
        A_hat = self._gen_matrix(rho)
        s = self._sample_vec(sigma, 0, self.eta1)
        e = self._sample_vec(sigma, self.k, self.eta1)
        s_hat = [_ntt(p) for p in s]
        e_hat = [_ntt(p) for p in e]
        As_hat = self._mat_vec_ntt(A_hat, s_hat)
        t_hat = [_poly_add(As_hat[i], e_hat[i]) for i in range(self.k)]
        ek = b"".join(_byte_encode(12, t_hat[i]) for i in range(self.k)) + rho
        dk = b"".join(_byte_encode(12, s_hat[i]) for i in range(self.k))
        return ek, dk, rho, sigma, s_hat, t_hat

    def encrypt_pke(self, ek: bytes, m: bytes, r: bytes):
        t_hat = [_byte_decode(12, ek[i*384:(i+1)*384]) for i in range(self.k)]
        rho = ek[384*self.k:]
        At_hat = self._gen_matrix(rho, transpose=True)
        r_vec = self._sample_vec(r, 0, self.eta1)
        e1 = self._sample_vec(r, self.k, self.eta2)
        e2 = _cbd(self.eta2, _prf(r, 2*self.k, 64*self.eta2))
        r_hat = [_ntt(p) for p in r_vec]
        u_hat = self._mat_vec_ntt(At_hat, r_hat)
        u = [_poly_add(_inv_ntt(u_hat[i]), e1[i]) for i in range(self.k)]
        mu = [_decompress(1, b) for b in _byte_decode(1, m)]
        inner = self._dot_ntt(t_hat, r_hat)
        v = _poly_add(_poly_add(_inv_ntt(inner), e2), mu)
        c1 = b"".join(_byte_encode(self.du, [_compress(self.du, c) for c in u[i]])
                      for i in range(self.k))
        c2 = _byte_encode(self.dv, [_compress(self.dv, c) for c in v])
        return c1 + c2

    def decrypt_pke(self, dk: bytes, c: bytes):
        s_hat = [_byte_decode(12, dk[i*384:(i+1)*384]) for i in range(self.k)]
        stride = 32 * self.du
        u = [[_decompress(self.du, v) for v in _byte_decode(self.du, c[i*stride:(i+1)*stride])]
             for i in range(self.k)]
        v = [_decompress(self.dv, v) for v in _byte_decode(self.dv, c[self.k*stride:])]
        u_hat = [_ntt(u[i]) for i in range(self.k)]
        inner = self._dot_ntt(s_hat, u_hat)
        w = _poly_sub(v, _inv_ntt(inner))
        return _byte_encode(1, [_compress(1, c) for c in w])

    def keygen(self, d: bytes = b"", z: bytes = b""):
        if not d: d = os.urandom(32)
        if not z: z = os.urandom(32)
        ek, dk_pke, *_ = self.keygen_pke(d)
        dk = dk_pke + ek + _H(ek) + z
        return ek, dk

    def encaps(self, ek: bytes, m: bytes = b""):
        if not m: m = os.urandom(32)
        k_bar, r = _G(m + _H(ek))
        c = self.encrypt_pke(ek, m, r)
        K = _J(k_bar + _H(c))
        return K, c

    def decaps(self, c: bytes, dk: bytes):
        dk_pke = dk[:self.dk_pke_size]
        ek = dk[self.dk_pke_size:self.dk_pke_size + self.ek_size]
        h = dk[self.dk_pke_size + self.ek_size:self.dk_pke_size + self.ek_size + 32]
        z = dk[self.dk_pke_size + self.ek_size + 32:]
        m_prime = self.decrypt_pke(dk_pke, c)
        k_bar_prime, r_prime = _G(m_prime + h)
        c_prime = self.encrypt_pke(ek, m_prime, r_prime)
        h_c = _H(c)
        diff = 0
        for a, b in zip(c, c_prime): diff |= a ^ b
        if diff == 0:
            return _J(k_bar_prime + h_c)
        else:
            return _J(z + h_c)
