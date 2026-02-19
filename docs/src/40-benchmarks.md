# Benchmarks

Performance measurements of CrystallographicFFT.jl across different space groups, grid sizes, and devices. CPU benchmarks use **single-threaded FFTW**; GPU benchmarks use **NVIDIA RTX 2080 Ti** (Float64). Times are median over multiple trials.

## CPU Benchmarks

### Cubic Groups (General KRFFT)

The general path (`GeneralCFFTPairPlan`) handles all 230 space groups. Here are representative cubic groups:

#### N = 64 (Full grid 64³)

| Group | SG | \|G\| | FFT (ms) | KRFFT (ms) | Speedup |
|-------|-----|------|----------|------------|---------|
| Pm-3m | 221 | 48 | 2.50 | 0.14 | **18×** |
| Pm-3 | 200 | 24 | 2.50 | 0.26 | **10×** |

#### N = 128 (Full grid 128³)

| Group | SG | \|G\| | FFT (ms) | KRFFT (ms) | Speedup |
|-------|-----|------|----------|------------|---------|
| Pm-3m | 221 | 48 | 34.5 | 1.22 | **28×** |
| Pm-3 | 200 | 24 | 34.5 | 2.12 | **16×** |

> **Note**: Speedup improves with grid size because the FFT cost grows as $O(N^3 \log N)$ while the KRFFT operates on a smaller subgrid.

### Centered Groups (Centered KRFFT)

The centered path (`CenteredCFFTPairPlan`) provides additional speedup by exploiting centering translations:

#### N = 64

| Group | SG | Centering | FFT (ms) | Forward (ms) | Backward (ms) | Speedup |
|-------|-----|-----------|----------|-------------|---------------|---------|
| Fm-3m | 225 | F | 2.50 | 0.13 | 0.20 | **19×** |
| Fd-3m | 227 | F | 2.50 | 0.13 | 0.20 | **19×** |
| Im-3m | 229 | I | 2.50 | 0.23 | 0.36 | **11×** |
| Ia-3d | 230 | I | 2.50 | 0.23 | 0.36 | **11×** |
| Fddd | 70 | F | 2.50 | 0.22 | 0.38 | **11×** |
| I4/mmm | 139 | I | 2.50 | 0.32 | 0.46 | **8×** |

#### N = 128

| Group | SG | Centering | FFT (ms) | Forward (ms) | Backward (ms) | Speedup |
|-------|-----|-----------|----------|-------------|---------------|---------|
| Fm-3m | 225 | F | 34.5 | 1.30 | 1.88 | **27×** |
| Fd-3m | 227 | F | 34.5 | 1.31 | 1.83 | **26×** |
| Im-3m | 229 | I | 34.5 | 2.30 | 3.20 | **15×** |

### Forward vs Backward Asymmetry

The backward transform is typically 1.3–1.7× slower than the forward:

| Direction | Reason |
|-----------|--------|
| Forward | Gather-based: sequential reads from Y(m), random writes to F̂(h) |
| Backward | Scatter-based: random reads from F̂(h), sequential writes to Y(m) |

This asymmetry is inherent to the reconstruction architecture and is consistent across all space groups.

### SCFT End-to-End Performance

Timing the complete diffusion step (forward → spectral multiply → backward) compared to full-grid FFT→K→IFFT:

#### N = 64

| Group | SG | Full FFT (ms) | SCFT (ms) | Speedup |
|-------|-----|--------------|-----------|---------|
| Pm-3m | 221 | 5.00 | 0.37 | **14×** |
| Fm-3m | 225 | 5.00 | 0.36 | **14×** |
| Im-3m | 229 | 5.00 | 0.65 | **8×** |

#### N = 128

| Group | SG | Full FFT (ms) | SCFT (ms) | Speedup |
|-------|-----|--------------|-----------|---------|
| Pm-3m | 221 | 69.0 | 2.83 | **24×** |
| Fm-3m | 225 | 69.0 | 3.27 | **21×** |

> **Note**: SCFT speedup includes both forward and backward transforms plus spectral multiplication. The full-grid baseline includes forward FFT + spectral multiply + inverse FFT.

---

## Real-Valued CFFT Benchmarks (`rcfft!` / `ircfft!`)

Using `rfft`/`irfft` instead of `fft`/`ifft` for real-valued fields. See [Real-Valued CFFT](@ref real-valued-cfft) for details.

### CPU rcfft vs cfft (N=128, M=64³)

| Group | \|G\| | cfft (μs) | rcfft (μs) | fwd× | icfft (μs) | ircfft (μs) | bwd× |
|-------|-----|-----------|------------|------|------------|-------------|------|
| Pm-3m | 48 | 3754 | 2038 | **1.84×** | 8483 | 3797 | **2.23×** |
| P4/mmm | 16 | 7061 | 4354 | **1.62×** | 14576 | 5711 | **2.55×** |
| P2/m | 4 | 19418 | 9866 | **1.97×** | 31111 | 8674 | **3.59×** |
| P-1 | 2 | 33346 | 18267 | **1.83×** | 46481 | 18907 | **2.46×** |

### GPU rcfft vs cfft (N=128, RTX 2080 Ti)

| Group | \|G\| | GPU rcfft (μs) | GPU cfft (μs) | fwd× | GPU ircfft (μs) | GPU icfft (μs) | bwd× |
|-------|------|---------------|---------------|------|----------------|----------------|------|
| Pm-3m | 48 | 175.9 | 521.1 | **2.96×** | 265.9 | 456.9 | **1.72×** |
| Fm-3m | 192 | 152.1 | 190.8 | **1.25×** | 256.5 | 436.2 | **1.70×** |
| Im-3m | 96 | 157.9 | 198.7 | **1.26×** | 260.7 | 437.7 | **1.68×** |

---

## GPU Benchmarks

GPU benchmarks measured on **NVIDIA GeForce RTX 2080 Ti** (Float64). See [GPU Acceleration](@ref gpu-acceleration) for full details.

### GPU CFFT Speedup (vs CUFFT N³)

| Group | \|G\| | Method | N=128 fwd | N=128 bwd | N=256 fwd | N=256 bwd |
|-------|------|--------|---------|---------|---------|-------|
| Pm-3m | 192 | general | 6.04× | 3.23× | **7.96×** | 2.39× |
| Pm-3 | 24 | general | 3.94× | 2.98× | 3.63× | 1.88× |
| Fm-3m | 192 | centered | 8.36× | 8.26× | **14.73×** | **8.59×** |
| Im-3m | 96 | centered | 6.08× | 5.70× | **8.48×** | **8.70×** |

### GPU vs CPU CFFT Absolute Acceleration

| Group | Method | N=128 | N=256 |
|-------|--------|-------|-------|
| Pm-3m | general | 15.5× | **34.4×** |
| Pm-3 | general | 19.2× | **35.4×** |
| Fm-3m | centered | 9.6× | **35.8×** |
| Im-3m | centered | 12.9× | **38.4×** |

### GPU Scaling (Fm-3m centered forward)

| N | Subgrid | GPU time (ms) | GPU speedup | Parity |
|---|---------|--------------|-------------|--------|
| 32 | 8³ | 0.058 | 0.65× | 4% |
| 64 | 16³ | 0.101 | 1.35× | 8% |
| 128 | 32³ | 0.148 | 8.2× | 35% |
| **256** | **64³** | **0.632** | **14.7×** | **64%** |

### Full-Grid FFT Baselines

| N | FFTW(N³) (ms) | CUFFT(N³) (ms) | FFTW/CUFFT |
|---|--------------|---------------|------------|
| 32 | 0.19 | 0.039 | 4.9× |
| 64 | 2.45 | 0.136 | 18× |
| 128 | 33 | 1.21 | 27× |
| 256 | 530 | 9.3 | 57× |

---

## Accuracy

All transforms achieve machine-precision accuracy:

| Metric | Value |
|--------|-------|
| Forward-backward roundtrip error | < 1e-14 |
| Spectral consistency (vs full FFT) | < 1e-12 |
| SCFT vs full-grid diffusion | < 1e-12 |
| GPU vs CPU forward | ~1e-14 |
| GPU vs CPU backward | ~1e-16 |
| rcfft roundtrip | < 1e-10 |
| rcfft vs cfft cross-validation | < 1e-12 |
