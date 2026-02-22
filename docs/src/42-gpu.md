# [GPU Acceleration](@id gpu-acceleration)

CrystallographicFFT.jl supports GPU acceleration via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and a CUDA extension. All plan types (`cfft!`, `icfft!`, `rcfft!`, `ircfft!`) work transparently on GPU with the same API.

## Setup

Install CUDA.jl alongside CrystallographicFFT.jl:

```julia
using Pkg
Pkg.add("CUDA")
```

Then load both packages and pass `array_type=CuArray{Float64}` to any plan constructor:

```julia
using CUDA
using CrystallographicFFT

N = (128, 128, 128)
pair = plan_cfft_pair(N, 225, 3; array_type=CuArray{Float64})

f0 = CuArray(randn(subgrid_size(pair)...))
F̂ = CUDA.zeros(ComplexF64, cfft_asu_size(pair))
f0_out = CUDA.zeros(Float64, subgrid_size(pair)...)

cfft!(F̂, pair, f0)
icfft!(f0_out, pair, F̂)
```

> **Note**: CUDA.jl is a *weak dependency* (package extension). Without CUDA.jl, the package works normally on CPU with zero overhead.

## Architecture

```
User API:  plan_cfft_pair(N, sg, dim; array_type=CuArray{Float64})
    │
    ├── ext/CUDAExt.jl: _infer_backend(CuArray) → CUDABackend()
    │
    ├── planning.jl: all buffers → _to_device(backend, ...)
    │                FFT plan on device arrays → CUFFT auto-dispatch
    │
    ├── execute.jl: backend dispatch
    │   ├── CPU: optimized @inbounds @simd scalar loops
    │   └── GPU: KernelAbstractions @kernel parallel kernels
    │
    └── cfft_api.jl: KA kernel copy_real_to_complex / copy_complex_to_real
```

### Design Principles

- **Plan-time on CPU**: spectral ASU enumeration, reconstruction table building, and matrix inversion all run on CPU. The results are transferred to GPU once via `_to_device`
- **Execute-time on GPU**: all transforms (`cfft!`, `icfft!`, etc.) run entirely on device with zero host-device transfers
- **Type parameterization**: plan structs use generic type parameters (`VA<:AbstractVector`, `P` for FFT plan type), so CPU `Vector` and GPU `CuVector` are handled uniformly
- **FFT dispatch**: `AbstractFFTs.plan_fft` on a `CuArray` automatically creates a CUFFT plan

## KA Kernel List

The following `@kernel` functions handle GPU execution:

| Kernel | Purpose |
|--------|---------|
| `reconstruct_general_kernel!` | Forward reconstruction (general path) |
| `reconstruct_rfft_kernel!` | Forward reconstruction (rfft path, signed indices) |
| `inv_reconstruct_fused_kernel!` | Backward reconstruction (fused, no F_work buffer) |
| `inv_reconstruct_rfft_kernel!` | Backward reconstruction (rfft path) |
| `assemble_g0_fused_kernel!` | G0 assembly with alive_mask (centered forward) |
| `assemble_g0_channel_kernel!` | G0 assembly per channel |
| `disassemble_g0_channel_kernel!` | G0 disassembly per channel (centered backward) |
| `centering_fold_2ch_kernel!` | Centering fold (2-channel, F lattice) |
| `centering_fold_4ch_kernel!` | Centering fold (4-channel, I/C/A lattice) |
| `centering_unfold_2ch_kernel!` | Centering unfold (2-channel) |
| `centering_unfold_4ch_kernel!` | Centering unfold (4-channel) |
| `orbit_gather_kernel!` | CSR orbit gather (centered backward) |
| `orbit_expand_kernel!` | Orbit phase expansion (centered backward) |
| `copy_real_to_complex_kernel!` | Real → Complex buffer copy |
| `reconstruct_rfft_kernel!` | Forward reconstruction (rfft path, signed indices) |
| `inv_reconstruct_rfft_kernel!` | Backward reconstruction (rfft path) |
| `copy_complex_to_real_kernel!` | Complex → Real buffer copy |

## GPU Optimizations

Two phases of GPU-specific optimizations have been implemented:

### Phase A: Sync Removal + Batched CUFFT

- **Removed all 14 redundant `synchronize` calls**: CUDA stream semantics guarantee ordering, making explicit device syncs unnecessary pure overhead (~15–50μs saved)
- **Batched CUFFT for centered path**: 4 independent CUFFT(H³) calls compressed into 1 batched CUFFT call, reducing kernel launch overhead by ~85μs

Result: Fm-3m centered forward **37% faster** (0.225ms → 0.141ms at N=128)

### Phase B: Kernel Fusion

- **Fused assemble_G0!**: merged `fill!(G0, 0)` + 4× per-channel assembly into a single kernel with `alive_mask` parity dispatch
- **Fused inv_reconstruct**: eliminated the `F_work = [F; conj(F)]` intermediate buffer by using signed index encoding — direct + conj access in a single kernel

Result: Fm-3m centered backward **46% faster** (0.300ms → 0.163ms at N=128)

## Performance

### GPU CFFT Speedup (vs CUFFT N³, Float64, RTX 2080 Ti)

#### Forward

| Group | \|G\| | Method | N=128 | N=256 |
|-------|------|--------|-------|-------|
| Pm-3m | 192 | general | 6.04× | **7.96×** |
| Pm-3 | 24 | general | 3.94× | 3.63× |
| Fm-3m | 192 | centered | 8.36× | **14.73×** |
| Im-3m | 96 | centered | 6.08× | **8.48×** |

#### Backward

| Group | \|G\| | Method | N=128 | N=256 |
|-------|------|--------|-------|-------|
| Pm-3m | 192 | general | 3.23× | 2.39× |
| Fm-3m | 192 | centered | 8.26× | **8.59×** |
| Im-3m | 96 | centered | 5.70× | **8.70×** |

### GPU vs CPU CFFT Absolute Acceleration

This is the most important metric for applications — how much faster is GPU CFFT than CPU CFFT:

| Group | Method | N=64 | N=128 | N=256 |
|-------|--------|------|-------|-------|
| Pm-3m | general | 3.0× | 15.5× | **34.4×** |
| Pm-3 | general | 4.7× | 19.2× | **35.4×** |
| Fm-3m | centered | 1.4× | 9.6× | **35.8×** |
| Im-3m | centered | 2.0× | 12.9× | **38.4×** |

> At N=256, GPU CFFT is **35–38× faster** than CPU CFFT across all tested groups.

### Absolute Timing (ms)

#### Forward

| Group | Method | N=128 CPU | N=128 GPU | N=256 CPU | N=256 GPU |
|-------|--------|-----------|-----------|-----------|-----------|
| Pm-3m | general | 3.04 | 0.202 | 45.0 | 1.306 |
| Fm-3m | centered | 1.38 | 0.148 | 22.6 | 0.632 |
| Im-3m | centered | 2.51 | 0.200 | 42.1 | 1.097 |

#### Backward

| Group | Method | N=128 CPU | N=128 GPU | N=256 CPU | N=256 GPU |
|-------|--------|-----------|-----------|-----------|-----------|
| Pm-3m | general | 7.49 | 0.425 | 90.4 | 3.670 |
| Fm-3m | centered | 1.82 | 0.163 | 25.8 | 0.891 |
| Im-3m | centered | 3.18 | 0.237 | 42.4 | 1.162 |

### Full-Grid FFT Baselines

| N | FFTW(N³) | CUFFT(N³) | FFTW/CUFFT |
|---|----------|-----------|------------|
| 32 | 0.19 ms | 0.039 ms | 4.9× |
| 64 | 2.45 ms | 0.136 ms | 18× |
| 128 | 33 ms | 1.21 ms | 27× |
| 256 | 530 ms | 9.3 ms | 57× |

## Scaling Behavior

GPU parity (GPU CFFT speedup / CPU CFFT speedup) improves with grid size:

| N | Subgrid M | Fm-3m GPU time | GPU speedup | Parity |
|---|-----------|---------------|-------------|--------|
| 32 | 8³ | 0.058 ms | 0.65× | 4% |
| 64 | 16³ | 0.101 ms | 1.35× | 8% |
| 128 | 32³ | 0.148 ms | 8.2× | 35% |
| **256** | **64³** | **0.632 ms** | **14.7×** | **64%** |

**Why parity < 100%**: GPU parity is limited by the CFFT algorithm's inherent overhead steps (fold, assemble, reconstruct) which are memory-bound scatter-gather operations. These steps are cache-friendly on CPU but suffer from irregular memory access on GPU. As N grows, the subgrid FFT dominates and parity improves.

## Accuracy

GPU transforms preserve machine-precision accuracy:

| Metric | Value |
|--------|-------|
| Forward (GPU vs CPU) | ~1e-14 |
| Backward (GPU vs CPU) | ~1e-16 |
| Roundtrip (GPU) | ~1e-14 |

All 12 GPU correctness tests pass (192 CPU tests unaffected).

## Recommendations

- **N ≥ 128** recommended for GPU acceleration — smaller grids may be slower than CUFFT(N³)
- **N = 256** is the sweet spot: parity 64–68%, absolute time < 1ms for forward transforms
- **Float64** is the default and recommended precision — Float32 does not improve GPU parity because CUFFT(N³) accelerates proportionally more
- The `centered` path benefits most from GPU optimization (batched CUFFT)
- For real-valued fields, `plan_rcfft_pair` with `array_type=CuArray{Float64}` combines rfft speedup with GPU acceleration
