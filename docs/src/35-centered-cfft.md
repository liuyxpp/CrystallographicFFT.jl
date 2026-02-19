# Centered CFFT

This page describes the **centered KRFFT path**, an optimized implementation for space groups with centering translations (I, C, A, F lattices). It provides additional speedup beyond the general path by exploiting translation symmetry within the subgrid.

## When Does This Path Apply?

Centering translations are fractional lattice translations that leave the crystal invariant:

| Centering | Translation vectors | Index |
|-----------|-------------------|-------|
| **I** (Body-centered) | $(½, ½, ½)$ | 2 |
| **C** (C-centered) | $(½, ½, 0)$ | 2 |
| **A** (A-centered) | $(0, ½, ½)$ | 2 |
| **F** (Face-centered) | $(½, ½, 0), (½, 0, ½), (0, ½, ½)$ | 4 |

The centering index $c$ determines how many independent sub-channels exist after folding.

## Plan Types

- **Forward**: `CFFTPlan{<:CenteredForwardPlan}` or the forward half of `CenteredCFFTPairPlan`
- **Backward**: `ICFFTPlan{<:CenteredBackwardPlan}` or the backward half of `CenteredCFFTPairPlan`

Plans are automatically created when `plan_cfft_pair` or `plan_cfft` detects a centered group. Use `method=:general` to force the general path instead:

```julia
plan_c = plan_cfft_pair(N, 225, 3)               # → CenteredCFFTPairPlan (auto)
plan_g = plan_cfft_pair(N, 225, 3; method=:general)  # → GeneralCFFTPairPlan (forced)
```

## Centering Fold Preprocessing

The key optimization: before applying the point-group reconstruction, the input subgrid is **folded** along the centering translations. For an F-centered group with stride-2 packing:

```
Input subgrid f₀(m)  [M³, M = N/2]
        │
        ▼
  ┌─────────────────────┐
  │  Centering Fold      │  Combine points related by centering
  └─────────────────────┘
        │
        ▼
  Folded channels  [c channels × (M/2)³]
```

The fold reduces the data volume by a factor of $c$ (the centering index). Each channel represents an even/odd parity combination under the centering translations.

## Multi-Channel FFT

After folding, a standard FFT is computed on each channel independently:

```
Channel 0: FFT on (M/2)³
Channel 1: FFT on (M/2)³
  ...
Channel c-1: FFT on (M/2)³
```

Since each channel is smaller than the original subgrid by a factor of $c$, the total FFT work is reduced, and FFTW batch planning can be applied.

## G0 Assembly

After the per-channel FFTs, the results are assembled into the spectral ASU using the remaining point-group operations (excluding the centering translations, which have already been exploited):

```
Per-channel FFT outputs
        │
        ▼
  ┌─────────────────────┐
  │  G0 Assembly         │  Point-group reconstruction on reduced spectrum
  └─────────────────────┘
        │
        ▼
  F̂(h)  [n_spec spectral ASU]
```

The G0 assembly uses the same dual-path (diagonal/general) architecture as the general KRFFT, but operates on a smaller dataset.

## Forward vs Backward for Centered Groups

| Step | Forward | Backward |
|------|---------|----------|
| 1 | Centering fold (real-space) | Inv. G0 assembly (spectral) |
| 2 | Per-channel FFT | Per-channel IFFT |
| 3 | G0 assembly (spectral) | Centering unfold (real-space) |

The backward transform reverses each step:
- **Inv. G0 Assembly**: scatters spectral ASU values back into per-channel spectra
- **Per-channel IFFT**: inverse FFT on each channel
- **Centering Unfold**: combines channels back into the full subgrid

## GPU Optimizations

On GPU, two key optimizations significantly improve centered path performance:

### Batched CUFFT

Instead of executing $c$ independent CUFFT calls sequentially, the channels are packed into a single 4D batch array and processed with one batched CUFFT call. This reduces kernel launch overhead from $4 \times \sim 30\,\mu s$ to a single $\sim 35\,\mu s$ call — a **37% speedup** for F-centered groups at N=128.

### Fused Kernels

- **Fused Assemble**: `fill!(G0, 0)` + per-channel assembly kernels are replaced by a single `assemble_g0_fused_kernel!` using an `alive_mask` to map each grid position to its channel
- **Fused Inv. Reconstruct**: the separate `F_work = [F; conj(F)]` buffer construction and scatter are merged into a single kernel using signed index encoding — saving both a kernel launch and the `F_work` buffer allocation. This gives a **46% speedup** on backward transforms for Fm-3m.

### GPU Parity and Scaling

GPU parity (GPU speedup / CPU speedup) for centered groups improves with grid size, as larger subgrids better utilize GPU parallelism:

| N | Subgrid | Fm-3m GPU speedup | Parity |
|---|---------|-------------------|--------|
| 32 | 8³ | 0.65× | 4% |
| 64 | 16³ | 1.35× | 8% |
| 128 | 32³ | 8.2× | 35% |
| **256** | **64³** | **14.7×** | **64%** |

N ≥ 128 is recommended for effective GPU acceleration.

## CPU Performance Impact

The centered path provides significant speedup for centered groups on CPU:

| Group | Centering | General speedup | Centered speedup | Improvement |
|-------|-----------|----------------|-----------------|-------------|
| Fm-3m (225) | F | ~6× | ~19× | 3× |
| Im-3m (229) | I | ~6× | ~11× | 1.8× |
| Fddd (70) | F | ~2× | ~11× | 5.5× |

The improvement is most dramatic for F-centered groups, where the centering index $c = 4$ gives a $4\times$ reduction in FFT work.
