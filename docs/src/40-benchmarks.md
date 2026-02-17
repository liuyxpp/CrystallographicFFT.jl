# Benchmarks

Performance measurements of CrystallographicFFT.jl across different space groups and grid sizes. All benchmarks use **single-threaded FFTW** for fair comparison. Times are median over multiple trials.

## Cubic Groups (General KRFFT)

The general path (`GeneralCFFTPairPlan`) handles all 230 space groups. Here are representative cubic groups:

### N = 64 (Full grid 64³)

| Group | SG | |G| | FFT (ms) | KRFFT (ms) | Speedup |
|-------|-----|------|----------|------------|---------|
| Pm-3m | 221 | 48 | 2.50 | 0.14 | **18×** |
| Pm-3 | 200 | 24 | 2.50 | 0.26 | **10×** |

### N = 128 (Full grid 128³)

| Group | SG | |G| | FFT (ms) | KRFFT (ms) | Speedup |
|-------|-----|------|----------|------------|---------|
| Pm-3m | 221 | 48 | 34.5 | 1.22 | **28×** |
| Pm-3 | 200 | 24 | 34.5 | 2.12 | **16×** |

> **Note**: Speedup improves with grid size because the FFT cost grows as $O(N^3 \log N)$ while the KRFFT operates on a smaller subgrid.

## Centered Groups (Centered KRFFT)

The centered path (`CenteredCFFTPairPlan`) provides additional speedup by exploiting centering translations:

### N = 64

| Group | SG | Centering | FFT (ms) | Forward (ms) | Backward (ms) | Speedup |
|-------|-----|-----------|----------|-------------|---------------|---------|
| Fm-3m | 225 | F | 2.50 | 0.13 | 0.20 | **19×** |
| Fd-3m | 227 | F | 2.50 | 0.13 | 0.20 | **19×** |
| Im-3m | 229 | I | 2.50 | 0.23 | 0.36 | **11×** |
| Ia-3d | 230 | I | 2.50 | 0.23 | 0.36 | **11×** |
| Fddd | 70 | F | 2.50 | 0.22 | 0.38 | **11×** |
| I4/mmm | 139 | I | 2.50 | 0.32 | 0.46 | **8×** |

### N = 128

| Group | SG | Centering | FFT (ms) | Forward (ms) | Backward (ms) | Speedup |
|-------|-----|-----------|----------|-------------|---------------|---------|
| Fm-3m | 225 | F | 34.5 | 1.30 | 1.88 | **27×** |
| Fd-3m | 227 | F | 34.5 | 1.31 | 1.83 | **26×** |
| Im-3m | 229 | I | 34.5 | 2.30 | 3.20 | **15×** |

## Forward vs Backward Asymmetry

The backward transform is typically 1.3–1.7× slower than the forward:

| Direction | Reason |
|-----------|--------|
| Forward | Gather-based: sequential reads from Y(m), random writes to F̂(h) |
| Backward | Scatter-based: random reads from F̂(h), sequential writes to Y(m) |

This asymmetry is inherent to the reconstruction architecture and is consistent across all space groups.

## SCFT End-to-End Performance

Timing the complete diffusion step (forward → spectral multiply → backward) compared to full-grid FFT→K→IFFT:

### N = 64

| Group | SG | Full FFT (ms) | SCFT (ms) | Speedup |
|-------|-----|--------------|-----------|---------|
| Pm-3m | 221 | 5.00 | 0.37 | **14×** |
| Fm-3m | 225 | 5.00 | 0.36 | **14×** |
| Im-3m | 229 | 5.00 | 0.65 | **8×** |

### N = 128

| Group | SG | Full FFT (ms) | SCFT (ms) | Speedup |
|-------|-----|--------------|-----------|---------|
| Pm-3m | 221 | 69.0 | 2.83 | **24×** |
| Fm-3m | 225 | 69.0 | 3.27 | **21×** |

> **Note**: SCFT speedup includes both forward and backward transforms plus spectral multiplication. The full-grid baseline includes forward FFT + spectral multiply + inverse FFT.

## Accuracy

All transforms achieve machine-precision accuracy:

| Metric | Value |
|--------|-------|
| Forward-backward roundtrip error | < 1e-14 |
| Spectral consistency (vs full FFT) | < 1e-12 |
| SCFT vs full-grid diffusion | < 1e-12 |
