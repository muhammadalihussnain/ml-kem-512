"""
ML-KEM-512 Performance Benchmark Script.

Run with: python3 scripts/benchmark.py

Measures KeyGen, Encaps, Decaps over N iterations and reports:
  - Min / Max / Mean time per operation
  - Key and ciphertext sizes
"""

import statistics
import time

from ml_kem_512.kem.decaps import decaps
from ml_kem_512.kem.encaps import encaps
from ml_kem_512.kem.keygen import keygen

N = 10  # iterations


def bench(label: str, fn, *args) -> list:
    times = []
    for _ in range(N):
        t = time.perf_counter()
        result = fn(*args)
        times.append((time.perf_counter() - t) * 1000)
    mean = statistics.mean(times)
    mn = min(times)
    mx = max(times)
    print(f"  {label:10s}  mean={mean:7.1f}ms  min={mn:7.1f}ms  max={mx:7.1f}ms")
    return result


def main():
    print(f"\nML-KEM-512 Benchmark ({N} iterations each)\n")

    d = bytes(range(32))
    z = bytes(range(1, 33))
    m = bytes([0x42] * 32)

    print("Timings:")
    ek, dk = bench("KeyGen", keygen, d, z)
    K1, c = bench("Encaps", encaps, ek, m)
    K2 = bench("Decaps", decaps, c, dk)

    print("\nSizes (ML-KEM-512 spec):")
    print(f"  ek (public key):      {len(ek):5d} bytes  (expected 800)")
    print(f"  dk (private key):     {len(dk):5d} bytes  (expected 1632)")
    print(f"  c  (ciphertext):      {len(c):5d} bytes  (expected 768)")
    print(f"  K  (shared secret):   {len(K1):5d} bytes  (expected 32)")

    print("\nCorrectness:")
    print(f"  K1 == K2: {K1 == K2}")
    print()


if __name__ == "__main__":
    main()
