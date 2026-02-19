```@meta
CurrentModule = CrystallographicFFT
```

# CrystallographicFFT.jl

Exploit crystallographic symmetry to compute FFTs on 3D periodic grids up to **28× faster** than standard FFTW — and up to **600× faster on GPU**.

## Introduction

CrystallographicFFT.jl implements the **KRFFT** (Kunis–Rössler FFT) algorithm for 3D grids with space-group symmetry. Instead of computing a full $N^3$-point FFT, the algorithm:

1. **Decomposes** the grid into a smaller subgrid (stride-$L$ sampling)
2. **Computes** a standard FFT on the subgrid ($M = N/L$ per dimension)
3. **Reconstructs** the Spectral ASU — the unique spectral coefficients under symmetry

The result is a compact spectral representation with $n_{\text{spec}} \approx N^3 / |G|$ unique coefficients, where $|G|$ is the order of the space group.

## Key Features

*   **⚡ Up to 28× speedup** over full-grid FFT for highly symmetric groups (Pm-3m, Fm-3m)
*   **🖥️ GPU acceleration** via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) — CUDA support through a lightweight extension; GPU CFFT is **35–38× faster** than CPU CFFT at N=256
*   **📐 Real-valued CFFT** (`rcfft!`/`ircfft!`) — uses `rfft`/`irfft` to halve FFT work for real fields, with 1.5–2× forward and 2–3× backward speedup
*   **🔬 All 230 space groups** via the general KRFFT path
*   **🧊 Centered lattice optimization** (I/C/A/F centering) with multi-channel fold for additional speedup
*   **🔄 Full forward + backward** transforms — forward (`cfft!`) and inverse (`icfft!`) with roundtrip error at machine precision
*   **🧪 SCFT-ready** — built-in diffusion kernel, `plan_cfft_pair` for symmetric boundary condition PDE solvers
*   **📦 Zero-copy plans** — pre-allocated buffers, no allocation during execution

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

### CPU

```julia
using CrystallographicFFT

N = (64, 64, 64)
sg_num = 225  # Fm-3m (face-centered cubic)

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

### GPU (CUDA)

```julia
using CUDA
using CrystallographicFFT

N = (64, 64, 64)
pair = plan_cfft_pair(N, 225, 3; array_type=CuArray{Float64})

f0 = CuArray(randn(subgrid_size(pair)...))
F̂ = CUDA.zeros(ComplexF64, cfft_asu_size(pair))
cfft!(F̂, pair, f0)

f0_out = CUDA.zeros(Float64, subgrid_size(pair)...)
icfft!(f0_out, pair, F̂)
```
