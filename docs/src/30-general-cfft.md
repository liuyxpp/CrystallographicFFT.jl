# General CFFT

This page describes the **general KRFFT path**, which works for all 230 space groups. It handles both forward (real-space → spectral) and backward (spectral → real-space) transforms.

## Plan Types

The general path produces plans of type:
- **Forward**: `CFFTPlan{<:ForwardPlan}` or the forward half of `GeneralCFFTPairPlan`
- **Backward**: `ICFFTPlan{<:BackwardPlan}` or the backward half of `GeneralCFFTPairPlan`

## Forward Data Flow

```
Real-space field f(x)  [N³ full grid, symmetric under G]
        │
        ▼
  ┌─────────────────┐
  │  Stride-L Pack   │  Extract every L-th point → subgrid
  └─────────────────┘
        │
        ▼
  f₀(m)  [M³ subgrid, M = N/L]
        │
        ▼
  ┌─────────────────┐
  │  Subgrid FFT     │  FFTW C2C on M³ (CPU) / CUFFT C2C on M³ (GPU)
  └─────────────────┘
        │
        ▼
  Y(m)  [M³ complex spectrum]
        │
        ▼
  ┌─────────────────┐
  │  Reconstruct     │  Sum over symmetry ops → spectral ASU
  └─────────────────┘
        │
        ▼
  F̂(h)  [n_spec unique coefficients]
```

## Backward Data Flow

```
  F̂(h)  [n_spec spectral ASU]
        │
        ▼
  ┌─────────────────────┐
  │  Inv. Reconstruct    │  Scatter from ASU → full subgrid spectrum
  └─────────────────────┘
        │
        ▼
  Y(m)  [M³ complex spectrum]
        │
        ▼
  ┌─────────────────┐
  │  Inverse FFT     │  FFTW C2C⁻¹ on M³ (CPU) / CUFFT C2C⁻¹ on M³ (GPU)
  └─────────────────┘
        │
        ▼
  f₀(m)  [M³ real-space subgrid]
```

## Dual-Path Reconstruction Architecture

The reconstruction step uses two internal paths:

### Diagonal Path
For frequencies where each symmetry operation maps to a unique subgrid index (no aliasing), reconstruction is a simple weighted sum:

```math
F(\mathbf{h}) = \sum_{g} \text{phase}_g \cdot Y[\text{idx}_g]
```

This is implemented as a direct table lookup — fast and cache-friendly.

### General Path
For frequencies with complex orbit structures (aliasing in the subgrid), reconstruction uses sparse matrix multiplication:

```math
F(\mathbf{h}) = \mathbf{w}^\top \cdot Y_{\text{gathered}}
```

where $\mathbf{w}$ is a precomputed sparse weight vector.

The plan construction automatically determines which path to use for each spectral ASU point. For high-symmetry cubic groups, the diagonal path handles the majority of frequencies.

## CPU vs GPU Execution

The algorithm is identical on CPU and GPU. The key differences are in the computational primitives:

| Step | CPU | GPU |
|------|-----|-----|
| Pack (stride-L extract) | `@inbounds @simd` loop | `copy_real_to_complex_kernel!` |
| Subgrid FFT | FFTW C2C | CUFFT C2C |
| Reconstruct | `@inbounds` SoA gather-reduce | `reconstruct_general_kernel!` |
| Inv. Reconstruct | `@inbounds` SoA scatter | `inv_reconstruct_fused_kernel!` |
| Inverse FFT | FFTW C2C⁻¹ | CUFFT C2C⁻¹ |

On CPU, the Pmmm fast path uses separable 8-point butterfly reconstruction for diagonal groups. On GPU, all groups use the general `reconstruct_general_kernel!` (the overhead of branching in the butterfly is higher than the benefit on GPU).

## Memory Layout

| Buffer | Size | Description |
|--------|------|-------------|
| Subgrid input | $M_1 \times M_2 \times M_3$ | Real-valued, extracted from full grid |
| Subgrid FFT | $M_1 \times M_2 \times M_3$ | Complex-valued (in-place or out-of-place) |
| Spectral ASU | $n_{\text{spec}}$ | Complex vector of unique coefficients |
| Reconstruction table | $n_{\text{ops}} \times n_{\text{spec}}$ | Precomputed (index, phase) pairs |

All buffers are pre-allocated during plan construction. Execution (`cfft!` / `icfft!`) is zero-allocation.
