# CrystallographicFFT.jl

[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://liuyxpp.github.io/CrystallographicFFT.jl/dev)
[![Test workflow status](https://github.com/liuyxpp/CrystallographicFFT.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/liuyxpp/CrystallographicFFT.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/liuyxpp/CrystallographicFFT.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/liuyxpp/CrystallographicFFT.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

Exploit crystallographic symmetry to compute FFTs on 3D periodic grids up to **28× faster** than standard FFTW — on both CPU and NVIDIA GPU.

## Key Features

*   **⚡ Up to 28× speedup** over full-grid FFT for highly symmetric groups (Pm-3m, Fm-3m)
*   **🔬 All 230 space groups** supported via the general KRFFT path
*   **🧊 Centered lattice optimization** (I/C/A/F centering) with multi-channel fold for additional speedup
*   **🖥️ GPU acceleration** — optional CUDA support via a zero-cost weak extension; just `using CUDA`
*   **� R2C FFT transforms** — `rcfft!`/`ircfft!` exploit real-valued symmetry for further memory and compute savings
*   **�🔄 Full forward + backward** transforms with roundtrip error at machine precision
*   **🌟 Star mapping** — `SubgridStarMap` for efficient symmetry-orbit conversions on the subgrid
*   **🧪 SCFT-ready** — built-in diffusion kernel, stress helpers, and `plan_cfft_pair` for symmetric PDE solvers
*   **📦 Zero-allocation execution** — pre-allocated plans, no GC pressure during transforms

## Installation

```julia
using Pkg
Pkg.add("CrystallographicFFT")
```

For GPU support, also install CUDA.jl:

```julia
Pkg.add("CUDA")
```

## Quick Example

### CPU (complex-valued)

```julia
using CrystallographicFFT

N = (64, 64, 64)
sg_num = 225  # Fm-3m (face-centered cubic, |G| = 192)

# Create a forward+backward plan
pair = plan_cfft_pair(N, sg_num, 3)

# Forward transform: subgrid → spectral ASU
f0 = rand(subgrid_size(pair)...)
F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
cfft!(F̂, pair, f0)

# Inverse transform: spectral ASU → subgrid
f0_out = zeros(subgrid_size(pair)...)
icfft!(f0_out, pair, F̂)

# Roundtrip at machine precision
@assert maximum(abs.(f0 .- f0_out)) < 1e-12
```

### CPU (real-valued, R2C)

```julia
using CrystallographicFFT

N = (64, 64, 64)
pair = plan_rcfft_pair(N, 221, 3)  # Pm-3m

f0 = rand(Float64, subgrid_size(pair)...)
F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))

rcfft!(F̂, pair, f0)     # real-valued forward
ircfft!(f0, pair, F̂)    # real-valued inverse
```

### GPU

```julia
using CUDA, CrystallographicFFT

N = (64, 64, 64)
pair = plan_cfft_pair(N, 225, 3; array_type=CuArray{Float64})

f0 = CUDA.rand(Float64, subgrid_size(pair)...)
F̂ = CuVector{ComplexF64}(undef, cfft_asu_size(pair))

cfft!(F̂, pair, f0)      # forward on GPU
icfft!(f0, pair, F̂)     # inverse on GPU
```

## How It Works

CrystallographicFFT.jl implements the **KRFFT** (Kunis–Rössler FFT) algorithm:

1. **Subgrid decomposition** — stride-$L$ sampling reduces the grid from $N^3$ to $(N/L)^3$
2. **Standard FFT** — FFTW (CPU) or cuFFT (GPU) on the smaller subgrid
3. **Symmetry reconstruction** — algebraic combination using space group operations to recover the unique spectral coefficients (Spectral ASU)

The result: $n_{\text{spec}} \approx N^3 / |G|$ unique coefficients instead of $N^3$.

The entire pipeline is **device-agnostic**, built on [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) — the same plan API works seamlessly on CPU and GPU.

## Performance

Representative benchmarks (single-threaded FFTW, N=128):

| Group | SG | Path | FFT (ms) | KRFFT (ms) | Speedup |
|-------|-----|------|----------|------------|---------|
| Pm-3m | 221 | General | 34.5 | 1.22 | **28×** |
| Fm-3m | 225 | Centered | 34.5 | 1.30 | **27×** |
| Im-3m | 229 | Centered | 34.5 | 2.30 | **15×** |
| Pm-3 | 200 | General | 34.5 | 2.12 | **16×** |

## Documentation

See the [full documentation](https://liuyxpp.github.io/CrystallographicFFT.jl/dev) for:

- [Tutorial & Usage](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/10-tutorial/) — plan API, SCFT workflow, utility functions
- [Theory & Algorithms](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/20-theory/) — KRFFT mathematical foundations
- [General CFFT](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/30-general-cfft/) — universal path for all 230 groups
- [Centered CFFT](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/35-centered-cfft/) — centering fold optimization
- [SCFT Integration](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/38-scft/) — diffusion solvers
- [R2C FFT](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/39-rcfft/) — real-valued transforms
- [Benchmarks](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/40-benchmarks/) — performance data
- [GPU Acceleration](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/42-gpu/) — CUDA setup and usage
- [API Reference](https://liuyxpp.github.io/CrystallographicFFT.jl/dev/95-reference/) — full API docs

## AI Usage Disclaimer

Most of source codes and docs in this project are generated with the help of Claude Opus 4.6 (thinking) and Gemini 3.0 Pro (High) in Google Antigravity. The LLM are guided by human with many rounds to achieve a pre-designed goal. And the AI generated contents are carefully examined by human. The correctness are verified with FFTW and the roundtrip transform. See `test` folder for verification details.
