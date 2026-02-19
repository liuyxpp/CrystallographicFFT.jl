# SCFT Integration

This page describes how CrystallographicFFT.jl integrates with **Self-Consistent Field Theory (SCFT)** solvers. The primary use case is solving the diffusion equation in spectral space for block copolymer and polymer blend systems.

## SCFT Diffusion Operator

In polymer SCFT, the core computation is applying the diffusion propagator:

```math
q(\mathbf{r}, s + \Delta s) = e^{\Delta s \, \nabla^2} \, q(\mathbf{r}, s)
```

In spectral space, this becomes a **pointwise multiplication**:

```math
\hat{q}(\mathbf{h}, s + \Delta s) = e^{-\Delta s \, |\mathbf{k}_h|^2} \, \hat{q}(\mathbf{h}, s)
```

where $|\mathbf{k}_h|^2$ is the squared wave vector of frequency $\mathbf{h}$.

## Using `plan_cfft_pair` for SCFT

The bidirectional pair plan is designed for this workflow:

```julia
using CrystallographicFFT

N = (64, 64, 64)
sg = 225          # Fm-3m
Δs = 0.01
lattice = [1.0 0 0; 0 1.0 0; 0 0 1.0]

pair = plan_cfft_pair(N, sg, 3)
K = make_diffusion_kernel(pair, Δs, lattice)

# Pre-allocate buffers
F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
f = rand(subgrid_size(pair)...)

# One diffusion step (forward → multiply → backward)
cfft!(F̂, pair, f)
@. F̂ *= K
icfft!(f, pair, F̂)
```

### Adaptive Step Size

When $\Delta s$ changes between iterations:

```julia
update_diffusion_kernel!(K, pair, new_Δs, lattice)
```

This is cheaper than creating a new kernel since it reuses the precomputed $|\mathbf{k}|^2$ values.

## Using Real-Valued Plans for SCFT

For real-valued fields, `rcfft!`/`ircfft!` provides additional speedup (1.5–2.5× over `cfft!`/`icfft!`):

```julia
rpair = plan_rcfft_pair(N, sg, 3)
K = make_diffusion_kernel(rpair, Δs, lattice)

F̂ = Vector{ComplexF64}(undef, cfft_asu_size(rpair))
f = rand(subgrid_size(rpair)...)

rcfft!(F̂, rpair, f)
@. F̂ *= K
ircfft!(f, rpair, F̂)
```

The real-valued path halves the FFT work and the backward reconstruction work. It supports all 230 space groups.

## GPU SCFT Workflow

For GPU-accelerated SCFT, simply pass `array_type=CuArray{Float64}`:

```julia
using CUDA
using CrystallographicFFT

N = (128, 128, 128)
sg = 225
Δs = 0.01
lattice = [1.0 0 0; 0 1.0 0; 0 0 1.0]

pair = plan_cfft_pair(N, sg, 3; array_type=CuArray{Float64})
K = make_diffusion_kernel(pair, Δs, lattice)

F̂ = CUDA.zeros(ComplexF64, cfft_asu_size(pair))
f = CuArray(randn(subgrid_size(pair)...))

# Same API — runs entirely on GPU
cfft!(F̂, pair, f)
@. F̂ *= K
icfft!(f, pair, F̂)
```

At N=256, the GPU SCFT step is **35–38× faster** than CPU. All plan types (`plan_cfft_pair`, `plan_rcfft_pair`) support GPU via the `array_type` keyword.

## Using Separate Forward/Backward Plans

For more control (e.g., when forward and backward are used in different contexts):

```julia
fwd = plan_cfft(N, sg, 3)
bwd = plan_icfft(fwd)    # efficient: shares geometry with fwd

K = make_diffusion_kernel(fwd, Δs, lattice)

cfft!(F̂, fwd, f)
@. F̂ *= K
icfft!(f, bwd, F̂)
```

## Spectral Utilities for SCFT

### Wave Vector Squared

Useful for custom spectral operators:

```julia
k2 = cfft_k2(plan, lattice)
# k2[i] = |k_i|² for each spectral ASU point
# k2[1] ≈ 0.0 (Γ point)
```

### Custom Spectral Operators

Build any spectral operator using `cfft_k2`:

```julia
# Poisson solver: Φ̂(h) = -ρ̂(h) / |k|²
k2 = cfft_k2(plan, lattice)
k2_inv = [k == 0 ? 0.0 : 1.0/k for k in k2]

cfft!(F̂, plan, ρ)
@. F̂ *= -k2_inv
icfft!(Φ, bwd, F̂)
```

## Performance Considerations

### CPU Performance

| Method | Best for | Typical speedup |
|--------|----------|----------------|
| `plan_cfft_pair` | High-symmetry groups (∣G∣ ≥ 16) | 10–27× |
| `plan_rcfft_pair` | All groups (real-valued fields) | +1.5–2.5× over cfft |

For most cubic space groups, `plan_cfft_pair` provides the best overall performance.

### GPU Performance (N=128, RTX 2080 Ti)

| Method | Fm-3m forward | Fm-3m backward | vs CPU |
|--------|--------------|----------------|--------|
| GPU `cfft!` (centered) | 0.148 ms | 0.163 ms | 9.6× |
| GPU `rcfft!` (general) | 0.152 ms | 0.257 ms | 10.8× |
| CUFFT(128³) baseline | 1.21 ms | — | — |

GPU speedup improves with grid size — at N=256, GPU CFFT is 35–38× faster than CPU CFFT.
