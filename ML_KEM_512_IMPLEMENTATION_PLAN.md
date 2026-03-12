# ML-KEM-512 Implementation Plan
## Complete Step-by-Step Guide with Testing Milestones

**Goal**: Implement ML-KEM-512 (FIPS 203) from scratch with verification at each step

**Strategy**: Build incrementally, test each component before moving forward

---

## Overview: What is ML-KEM-512?

ML-KEM (Module-Lattice-Based Key Encapsulation Mechanism) is the NIST-standardized version of CRYSTALS-Kyber.

**ML-KEM-512 Parameters**:
- Security Level: NIST Level 1 (~128-bit security)
- Polynomial degree: n = 256
- Modulus: q = 3329
- Module rank: k = 2
- Polynomial ring: R_q = Z_q[x]/(x^256 + 1)
- Error distribution: CBD with η₁ = 3, η₂ = 2

**Three Main Operations**:
1. **KeyGen**: Generate public/private key pair
2. **Encaps**: Encapsulate a shared secret
3. **Decaps**: Decapsulate to recover shared secret

---

## Implementation Phases

```
Phase 1: Foundation (Primitives)          [Week 1]
Phase 2: Polynomial Arithmetic            [Week 1-2]
Phase 3: Sampling & Compression           [Week 2]
Phase 4: Module Operations                [Week 2-3]
Phase 5: Key Generation                   [Week 3]
Phase 6: Encapsulation                    [Week 3-4]
Phase 7: Decapsulation                    [Week 4]
Phase 8: Integration & Testing            [Week 4]
Phase 9: Optimization (Optional)          [Week 5+]
```

---

## PHASE 1: Foundation (Cryptographic Primitives)

### Milestone 1.1: Hash Functions (SHAKE, SHA3)
**What to implement**:
- SHAKE-128 (extendable output function)
- SHAKE-256 (extendable output function)
- SHA3-256 (for hashing)
- SHA3-512 (for hashing)

**Why needed**: ML-KEM uses these for:
- Deterministic randomness expansion
- Hashing public keys
- Key derivation

**Implementation approach**:
- Use Python's `hashlib` library (already implements FIPS 202)
- Create wrapper functions for ML-KEM specific usage

**Test criteria**:
```
✓ SHAKE-128 produces correct output for test vectors
✓ SHAKE-256 produces correct output for test vectors
✓ SHA3-256 produces correct output for test vectors
✓ SHA3-512 produces correct output for test vectors
✓ Can generate arbitrary-length output from SHAKE
```

**Test vectors**: Use NIST FIPS 202 test vectors

---

### Milestone 1.2: Pseudorandom Functions (PRF)
**What to implement**:
- PRF(s, b): SHAKE-256 based PRF
- XOF(ρ, i, j): Extendable output function for matrix generation

**Why needed**: 
- Generate pseudorandom bytes for sampling
- Expand seeds into matrix elements

**Test criteria**:
```
✓ PRF produces deterministic output from seed
✓ XOF generates correct matrix elements
✓ Output matches reference implementation
```

---

### Milestone 1.3: Key Derivation Functions
**What to implement**:
- H(x): SHA3-256 hash
- J(x): SHA3-512 hash  
- G(x): SHA3-512 for key derivation

**Test criteria**:
```
✓ Hash functions produce correct output
✓ Can hash arbitrary-length inputs
✓ Output length is correct (256 or 512 bits)
```

---

## PHASE 2: Polynomial Arithmetic

### Milestone 2.1: Basic Polynomial Operations
**What to implement**:
- Polynomial representation (array of 256 coefficients)
- Polynomial addition in R_q
- Polynomial subtraction in R_q
- Coefficient-wise modular reduction

**Test criteria**:
```
✓ Can create polynomial with 256 coefficients
✓ Addition works correctly (mod q)
✓ Subtraction works correctly (mod q)
✓ All coefficients stay in range [0, q-1]
✓ Test with known examples
```

**Example test**:
```
p1 = [1, 2, 3, ..., 0]
p2 = [4, 5, 6, ..., 0]
p3 = p1 + p2 (mod q)
Verify: p3[0] = 5, p3[1] = 7, etc.
```

---

### Milestone 2.2: Polynomial Multiplication (Schoolbook)
**What to implement**:
- Schoolbook multiplication
- Reduction modulo (x^256 + 1)
- Handle negative wraparound: x^256 ≡ -1

**Test criteria**:
```
✓ Multiply two polynomials correctly
✓ Reduction modulo (x^256 + 1) works
✓ x * x = x^2
✓ x^256 = -1 ≡ q-1 (mod q)
✓ Commutative: a*b = b*a
✓ Associative: (a*b)*c = a*(b*c)
```

**Critical test**:
```
x = [0, 1, 0, ..., 0]  # polynomial x
Compute x^256 by repeated multiplication
Result should be [q-1, 0, 0, ..., 0]  # -1 mod q
```

---

### Milestone 2.3: Number Theoretic Transform (NTT)
**What to implement**:
- Forward NTT (polynomial → NTT domain)
- Inverse NTT (NTT domain → polynomial)
- NTT-based multiplication (much faster)
- Precompute twiddle factors (roots of unity)

**Why needed**: NTT makes multiplication O(n log n) instead of O(n²)

**Parameters for ML-KEM-512**:
- n = 256
- q = 3329
- Root of unity: ζ = 17 (primitive 512-th root of unity mod q)
- Bit-reversal permutation

**Test criteria**:
```
✓ Forward NTT transforms correctly
✓ Inverse NTT recovers original polynomial
✓ NTT(a) * NTT(b) = NTT(a*b) (component-wise)
✓ InvNTT(NTT(a)) = a
✓ NTT multiplication matches schoolbook result
✓ Performance: NTT is faster than schoolbook
```

**Test with known vectors**:
```
Use test vectors from Kyber reference implementation
Compare NTT output with reference
```

---

## PHASE 3: Sampling & Compression

### Milestone 3.1: Centered Binomial Distribution (CBD)
**What to implement**:
- CBD_η(PRF(s, N)): Sample from centered binomial distribution
- η₁ = 3 (for secrets in key generation)
- η₂ = 2 (for errors and encapsulation)

**Algorithm**:
```
CBD_η(bytes):
  For each coefficient:
    a = sum of η random bits
    b = sum of η random bits
    coefficient = a - b
```

**Test criteria**:
```
✓ Samples 256 coefficients
✓ Coefficients in range [-η, η]
✓ Distribution is centered (mean ≈ 0)
✓ Deterministic from seed
✓ Statistical tests pass (chi-square test)
```

**Statistical test**:
```
Generate 10,000 samples
Check distribution matches expected binomial
Mean should be ≈ 0
Variance should be ≈ η/2
```

---

### Milestone 3.2: Uniform Sampling
**What to implement**:
- Sample uniform random polynomial from XOF
- Rejection sampling to ensure uniform distribution
- Parse bytes into coefficients mod q

**Algorithm**:
```
SampleNTT(XOF(ρ, i, j)):
  Generate bytes from XOF
  Parse into 12-bit values
  Reject if >= q
  Continue until 256 coefficients
```

**Test criteria**:
```
✓ Generates 256 coefficients
✓ All coefficients in [0, q-1]
✓ Distribution is uniform
✓ Deterministic from seed (ρ, i, j)
✓ Matches reference implementation
```

---

### Milestone 3.3: Compression & Decompression
**What to implement**:
- Compress_d(x): Compress coefficient to d bits
- Decompress_d(x): Decompress back to Z_q
- Used to reduce ciphertext size

**Formulas**:
```
Compress_d(x) = ⌊(2^d / q) * x⌉ mod 2^d
Decompress_d(x) = ⌊(q / 2^d) * x⌉
```

**ML-KEM-512 uses**:
- d_u = 10 (for vector u)
- d_v = 4 (for value v)

**Test criteria**:
```
✓ Compress reduces coefficient size
✓ Decompress(Compress(x)) ≈ x (small error)
✓ Error is bounded: |x - Decompress(Compress(x))| < q/(2^(d+1))
✓ Test with boundary values (0, q-1, q/2)
```

---

### Milestone 3.4: Encoding & Decoding
**What to implement**:
- ByteEncode_d(polynomial): Encode to bytes
- ByteDecode_d(bytes): Decode from bytes
- Bit packing for efficient storage

**Test criteria**:
```
✓ Encode produces correct byte length
✓ Decode(Encode(p)) = p
✓ Handles all coefficient values
✓ Bit packing is correct
```

---

## PHASE 4: Module Operations

### Milestone 4.1: Vector Operations
**What to implement**:
- Vector of k polynomials (k=2 for ML-KEM-512)
- Vector addition
- Vector subtraction
- Component-wise operations

**Test criteria**:
```
✓ Can create vector of 2 polynomials
✓ Addition works component-wise
✓ Subtraction works component-wise
```

---

### Milestone 4.2: Matrix-Vector Multiplication
**What to implement**:
- Matrix A: k×k matrix of polynomials
- Matrix-vector product: A · s
- Use NTT for efficiency

**Algorithm**:
```
For each row i:
  result[i] = sum(A[i][j] * s[j] for j in 0..k-1)
```

**Test criteria**:
```
✓ Matrix multiplication is correct
✓ Result is vector of k polynomials
✓ Matches schoolbook multiplication
✓ NTT version matches non-NTT version
```

---

### Milestone 4.3: Dot Product
**What to implement**:
- Dot product of two vectors
- t^T · s (transpose times vector)

**Test criteria**:
```
✓ Dot product produces single polynomial
✓ Correct for known test vectors
```

---

## PHASE 5: Key Generation

### Milestone 5.1: Matrix Generation
**What to implement**:
```
K-PKE.KeyGen():
  Input: random seed d (32 bytes)
  
  Step 1: Expand seed
    (ρ, σ) = G(d)  // ρ: 32 bytes, σ: 32 bytes
  
  Step 2: Generate matrix A
    For i in 0..k-1:
      For j in 0..k-1:
        A[i][j] = SampleNTT(XOF(ρ, j, i))  // Note: j, i order
  
  Output: Matrix A (in NTT domain)
```

**Test criteria**:
```
✓ Matrix A is k×k (2×2 for ML-KEM-512)
✓ Each element is polynomial in NTT domain
✓ Deterministic from seed ρ
✓ Matches reference implementation
```

---

### Milestone 5.2: Secret Vector Generation
**What to implement**:
```
  Step 3: Sample secret vector s
    N = 0
    For i in 0..k-1:
      s[i] = CBD_η₁(PRF(σ, N))
      N = N + 1
    
  Step 4: Transform to NTT domain
    ŝ = NTT(s)
```

**Test criteria**:
```
✓ Secret vector has k polynomials
✓ Each polynomial sampled from CBD_η₁
✓ Coefficients in range [-η₁, η₁]
✓ NTT transformation correct
```

---

### Milestone 5.3: Error Vector Generation
**What to implement**:
```
  Step 5: Sample error vector e
    For i in 0..k-1:
      e[i] = CBD_η₁(PRF(σ, N))
      N = N + 1
    
  Step 6: Transform to NTT domain
    ê = NTT(e)
```

**Test criteria**:
```
✓ Error vector has k polynomials
✓ Each polynomial sampled from CBD_η₁
✓ Coefficients in range [-η₁, η₁]
```

---

### Milestone 5.4: Public Key Computation
**What to implement**:
```
  Step 7: Compute public key
    t̂ = A ∘ ŝ + ê  // ∘ is matrix-vector mult in NTT domain
  
  Step 8: Encode keys
    pk = ByteEncode₁₂(t̂) || ρ  // 384*k + 32 bytes
    sk = ByteEncode₁₂(ŝ)        // 384*k bytes
```

**Test criteria**:
```
✓ Public key t̂ computed correctly
✓ t̂ = A·s + e (verify in polynomial domain)
✓ Public key encoding correct
✓ Secret key encoding correct
✓ Key sizes match specification:
  - pk: 800 bytes for ML-KEM-512
  - sk: 768 bytes for ML-KEM-512
```

---

### Milestone 5.5: Complete KeyGen
**What to implement**:
```
ML-KEM.KeyGen():
  d ← {0,1}^256  // 32 random bytes
  (ekₚₖₑ, dkₚₖₑ) ← K-PKE.KeyGen(d)
  
  ek = ekₚₖₑ  // encapsulation key (public key)
  dk = (dkₚₖₑ || ek || H(ek) || z)  // decapsulation key
  
  where z ← {0,1}^256  // 32 random bytes
```

**Test criteria**:
```
✓ Encapsulation key (public key) is 800 bytes
✓ Decapsulation key (private key) is 1632 bytes
✓ Keys are deterministic from seed d
✓ Can generate multiple key pairs
✓ Each key pair is different (with different seeds)
```

---

## PHASE 6: Encapsulation

### Milestone 6.1: Message Encoding
**What to implement**:
```
Encaps(ek):
  Step 1: Generate random message
    m ← {0,1}^256  // 32 random bytes
  
  Step 2: Hash inputs
    (K̄, r) = G(m || H(ek))  // K̄: 32 bytes, r: 32 bytes
```

**Test criteria**:
```
✓ Random message generated
✓ Hash computed correctly
✓ K̄ and r have correct lengths
```

---

### Milestone 6.2: Encryption (K-PKE.Encrypt)
**What to implement**:
```
K-PKE.Encrypt(ek, m, r):
  Step 1: Decode public key
    (t̂, ρ) = ek
    t̂ = ByteDecode₁₂(t̂)
  
  Step 2: Regenerate matrix A
    A = Gen(ρ)  // Same as in KeyGen
  
  Step 3: Sample random vector r
    N = 0
    For i in 0..k-1:
      r[i] = CBD_η₁(PRF(r, N))
      N = N + 1
    r̂ = NTT(r)
  
  Step 4: Sample error vectors
    For i in 0..k-1:
      e1[i] = CBD_η₂(PRF(r, N))
      N = N + 1
    e2 = CBD_η₂(PRF(r, N))
  
  Step 5: Compute ciphertext
    u = InvNTT(Aᵀ ∘ r̂) + e1
    μ = Decompress₁(ByteDecode₁(m))
    v = InvNTT(t̂ᵀ ∘ r̂) + e2 + μ
  
  Step 6: Compress and encode
    c1 = ByteEncode_{d_u}(Compress_{d_u}(u))
    c2 = ByteEncode_{d_v}(Compress_{d_v}(v))
    c = c1 || c2
```

**Test criteria**:
```
✓ Matrix A regenerated correctly
✓ Random vector r sampled correctly
✓ Error vectors e1, e2 sampled correctly
✓ u computed correctly
✓ v computed correctly
✓ Ciphertext size correct: 768 bytes for ML-KEM-512
✓ Encryption is deterministic from r
```

---

### Milestone 6.3: Complete Encapsulation
**What to implement**:
```
ML-KEM.Encaps(ek):
  m ← {0,1}^256
  (K̄, r) = G(m || H(ek))
  c = K-PKE.Encrypt(ek, m, r)
  K = J(K̄ || H(c))  // Final shared secret
  
  return (K, c)  // K: 32 bytes, c: 768 bytes
```

**Test criteria**:
```
✓ Shared secret K is 32 bytes
✓ Ciphertext c is 768 bytes
✓ Different m produces different K and c
✓ Same m and ek produce same K and c
```

---

## PHASE 7: Decapsulation

### Milestone 7.1: Decryption (K-PKE.Decrypt)
**What to implement**:
```
K-PKE.Decrypt(dk, c):
  Step 1: Decode secret key and ciphertext
    ŝ = ByteDecode₁₂(dk)
    (u, v) = c
    u = Decompress_{d_u}(ByteDecode_{d_u}(u))
    v = Decompress_{d_v}(ByteDecode_{d_v}(v))
  
  Step 2: Compute message
    û = NTT(u)
    w = v - InvNTT(ŝᵀ ∘ û)
    m = ByteEncode₁(Compress₁(w))
  
  return m
```

**Test criteria**:
```
✓ Decryption recovers message m
✓ Decrypt(Encrypt(m)) = m
✓ Works with all test vectors
✓ Small errors don't cause failure
```

---

### Milestone 7.2: Complete Decapsulation
**What to implement**:
```
ML-KEM.Decaps(c, dk):
  Step 1: Parse decapsulation key
    (dkₚₖₑ, ekₚₖₑ, h, z) = dk
  
  Step 2: Decrypt ciphertext
    m' = K-PKE.Decrypt(dkₚₖₑ, c)
  
  Step 3: Re-encrypt to verify
    (K̄', r') = G(m' || h)
    c' = K-PKE.Encrypt(ekₚₖₑ, m', r')
  
  Step 4: Check if ciphertext matches
    if c = c':
      K = J(K̄' || H(c))  // Success
    else:
      K = J(z || H(c))    // Failure (implicit rejection)
  
  return K
```

**Test criteria**:
```
✓ Decapsulation recovers correct shared secret
✓ Decaps(Encaps(ek)) produces same K
✓ Modified ciphertext produces different K (implicit rejection)
✓ Invalid ciphertext doesn't crash
```

---

## PHASE 8: Integration & Testing

### Milestone 8.1: End-to-End Test
**What to test**:
```
Test 1: Basic functionality
  (ek, dk) = KeyGen()
  (K1, c) = Encaps(ek)
  K2 = Decaps(c, dk)
  assert K1 == K2

Test 2: Multiple encapsulations
  Generate 100 key pairs
  For each: encapsulate and decapsulate
  All should succeed

Test 3: Different messages
  Same key pair, different messages
  Should produce different ciphertexts and keys
```

**Test criteria**:
```
✓ KeyGen → Encaps → Decaps works
✓ Shared secret matches
✓ 100% success rate on valid inputs
```

---

### Milestone 8.2: NIST Test Vectors
**What to test**:
- Download official ML-KEM test vectors from NIST
- Test KeyGen with known seeds
- Test Encaps with known randomness
- Test Decaps with known ciphertexts

**Test criteria**:
```
✓ KeyGen output matches NIST vectors
✓ Encaps output matches NIST vectors
✓ Decaps output matches NIST vectors
✓ All official test vectors pass
```

---

### Milestone 8.3: Negative Tests
**What to test**:
```
Test 1: Corrupted ciphertext
  Flip random bits in ciphertext
  Decaps should produce different K (implicit rejection)

Test 2: Wrong key
  Encaps with one key, Decaps with another
  Should produce different K

Test 3: Invalid inputs
  Test with wrong sizes, invalid values
  Should handle gracefully
```

**Test criteria**:
```
✓ Corrupted ciphertext handled correctly
✓ No crashes on invalid input
✓ Implicit rejection works
```

---

### Milestone 8.4: Performance Benchmarks
**What to measure**:
```
- KeyGen time
- Encaps time
- Decaps time
- Memory usage
- Key sizes
- Ciphertext size
```

**Target performance** (Python, unoptimized):
- KeyGen: < 5ms
- Encaps: < 5ms
- Decaps: < 5ms

**Test criteria**:
```
✓ Performance is reasonable
✓ No memory leaks
✓ Sizes match specification
```

---

## PHASE 9: Optimization (Optional)

### Milestone 9.1: NTT Optimization
- Precompute twiddle factors
- Use Barrett reduction
- Optimize bit-reversal

### Milestone 9.2: Constant-Time Implementation
- Remove branches based on secret data
- Constant-time comparison
- Protect against timing attacks

### Milestone 9.3: Vectorization
- Use SIMD instructions
- Batch operations
- Parallel NTT

---

## Testing Strategy

### Unit Tests (Test Each Component)
```
✓ Hash functions
✓ PRF and XOF
✓ Polynomial operations
✓ NTT forward/inverse
✓ CBD sampling
✓ Compression/decompression
✓ Encoding/decoding
✓ Matrix operations
```

### Integration Tests (Test Combinations)
```
✓ KeyGen components together
✓ Encaps components together
✓ Decaps components together
```

### System Tests (Test Complete Flow)
```
✓ Full KeyGen → Encaps → Decaps
✓ NIST test vectors
✓ Negative tests
✓ Performance tests
```

---

## File Structure

```
ml-kem-512/
├── src/
│   ├── __init__.py
│   ├── parameters.py          # ML-KEM-512 parameters
│   ├── primitives.py          # Hash functions, PRF, XOF
│   ├── polynomial.py          # Polynomial operations
│   ├── ntt.py                 # Number Theoretic Transform
│   ├── sampling.py            # CBD, uniform sampling
│   ├── compression.py         # Compress/decompress
│   ├── encoding.py            # Byte encode/decode
│   ├── module.py              # Vector/matrix operations
│   ├── pke.py                 # K-PKE (encryption scheme)
│   └── kem.py                 # ML-KEM (full KEM)
├── tests/
│   ├── test_primitives.py
│   ├── test_polynomial.py
│   ├── test_ntt.py
│   ├── test_sampling.py
│   ├── test_compression.py
│   ├── test_encoding.py
│   ├── test_module.py
│   ├── test_pke.py
│   ├── test_kem.py
│   └── test_vectors/          # NIST test vectors
├── examples/
│   ├── demo_keygen.py
│   ├── demo_encaps.py
│   └── demo_full.py
└── docs/
    ├── IMPLEMENTATION_GUIDE.md
    └── TESTING_GUIDE.md
```

---

## Implementation Order (Recommended)

### Week 1: Foundation
1. Day 1-2: Primitives (hash functions, PRF, XOF)
2. Day 3-4: Polynomial operations (add, subtract, multiply)
3. Day 5-7: NTT implementation and testing

### Week 2: Sampling & Encoding
1. Day 1-2: CBD sampling
2. Day 3-4: Compression/decompression
3. Day 5-7: Encoding/decoding, module operations

### Week 3: Key Generation & Encapsulation
1. Day 1-3: Complete KeyGen
2. Day 4-7: Complete Encaps

### Week 4: Decapsulation & Testing
1. Day 1-3: Complete Decaps
2. Day 4-7: Integration testing, NIST vectors

---

## Success Criteria

ML-KEM-512 implementation is complete when:

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All NIST test vectors pass
- [ ] KeyGen → Encaps → Decaps works correctly
- [ ] Shared secret matches between parties
- [ ] Implicit rejection works
- [ ] Performance is acceptable
- [ ] Code is documented
- [ ] You understand every component

---

## After ML-KEM-512

Once ML-KEM-512 is working:

**ML-KEM-768** (just change parameters):
- k = 3 (instead of 2)
- η₁ = 2 (instead of 3)
- Same algorithms, different sizes

**ML-KEM-1024** (just change parameters):
- k = 4 (instead of 2)
- η₁ = 2 (instead of 3)
- Same algorithms, different sizes

The hard work is ML-KEM-512. The others are parameter changes!

---

## Resources

**Official Specification**:
- FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard
- https://csrc.nist.gov/pubs/fips/203/final

**Reference Implementation**:
- CRYSTALS-Kyber: https://github.com/pq-crystals/kyber
- PQClean: https://github.com/PQClean/PQClean

**Test Vectors**:
- NIST: https://csrc.nist.gov/Projects/post-quantum-cryptography

---

**Ready to start? Begin with Phase 1, Milestone 1.1!**
