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

## Internal SCFT Paths

For advanced users, two internal SCFT execution paths exist (accessible via `CrystallographicFFT.QFusedKRFFT`):

### M2 Forward+Backward SCFT

```julia
using CrystallographicFFT.QFusedKRFFT: plan_m2_scft, execute_m2_scft!

scft = plan_m2_scft(N, sg, 3, Δs, lattice)
execute_m2_scft!(scft, f)  # in-place: f₀ → F̂·K → f₀
```

This bundles the forward+multiply+backward into a single call with optimal buffer reuse.

### Q-Fused SCFT

```julia
using CrystallographicFFT.QFusedKRFFT: plan_m2_q, execute_m2_q!

m2q = plan_m2_q(N, sg, 3, Δs, lattice)
execute_m2_q!(m2q, f)  # Q-matrix fused: FFT → Q·Y → IFFT
```

The Q-fused path replaces the separate reconstruct+multiply+inv_reconstruct steps with a single $Q$-matrix multiplication in the subgrid spectral domain. This can be faster for low-symmetry groups where the spectral ASU is large.

## Performance Considerations

| Method | Best for | Typical speedup |
|--------|----------|----------------|
| `plan_cfft_pair` | High-symmetry groups (|G| ≥ 16) | 10–27× |
| `plan_m2_scft` | Mid-symmetry groups (8 ≤ |G| < 16) | 5–10× |
| `plan_m2_q` | Low-symmetry groups (|G| < 8) | 2–5× |

For most cubic space groups, `plan_cfft_pair` provides the best overall performance.
