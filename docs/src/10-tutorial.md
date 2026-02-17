# Tutorial & Usage

## Getting Started

```julia
using CrystallographicFFT
using FFTW  # Required: provides the underlying FFT engine
```

The two key inputs for any CFFT plan are:
- **`N`**: the full-grid dimensions as a tuple, e.g., `(64, 64, 64)`
- **`sg_num`**: the space group number (1–230), e.g., `225` for Fm-3m

## Bidirectional Plans (`plan_cfft_pair`)

When you need both forward and backward transforms (e.g., PDE solvers, SCFT), use a **pair plan**:

```julia
N = (64, 64, 64)
pair = plan_cfft_pair(N, 225, 3)  # Fm-3m, 3D
```

The plan automatically selects the optimal path:
- **Centered groups** (I/C/A/F lattice) → `CenteredCFFTPairPlan` (exploits centering fold)
- **All other groups** → `GeneralCFFTPairPlan` (universal KRFFT)

You can force the general path with `method=:general`:

```julia
pair_gen = plan_cfft_pair(N, 225, 3; method=:general)
```

### Forward Transform

```julia
M = subgrid_size(pair)       # e.g., (32, 32, 32)
f0 = rand(M...)              # real-space field on the subgrid
F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))

cfft!(F̂, pair, f0)          # in-place forward: f0 → F̂
```

### Inverse Transform

```julia
f0_out = zeros(M...)
icfft!(f0_out, pair, F̂)     # in-place inverse: F̂ → f0_out
```

---

## Single-Direction Plans

When you only need one direction, use separate forward and backward plans:

### Forward Only

```julia
fwd = plan_cfft(N, 225, 3)           # → CFFTPlan
cfft!(F̂, fwd, f0)
```

### Backward from Forward

The most efficient way to create a backward plan — reuses geometry computed in the forward plan:

```julia
bwd = plan_icfft(fwd)                # → ICFFTPlan (shares geometry)
icfft!(f0_out, bwd, F̂)
```

### Standalone Backward

```julia
bwd = plan_icfft(N, 225, 3)          # → ICFFTPlan (independent)
```

---

## SCFT Diffusion Step

A common use case: solving the diffusion equation in spectral space.

### Building the Diffusion Kernel

```julia
Δs = 0.05                            # contour step
lattice = [1.0 0 0; 0 1.0 0; 0 0 1.0]  # unit cell (identity = cubic)

pair = plan_cfft_pair(N, 225, 3)
K = make_diffusion_kernel(pair, Δs, lattice)
# K[i] = exp(-Δs * |k_i|²) for each spectral ASU point
```

### Diffusion Execution

```julia
F̂ = Vector{ComplexF64}(undef, cfft_asu_size(pair))
cfft!(F̂, pair, f0)
@. F̂ *= K
icfft!(f0, pair, F̂)
```

### Updating the Kernel

When `Δs` changes between iterations (e.g., adaptive stepping):

```julia
update_diffusion_kernel!(K, pair, new_Δs, lattice)
```

---

## Grid Size Selection

Choose grid sizes that are compatible with your space group:

```julia
N_good = recommended_N(64, 225)  # → a size ≥ 64 compatible with SG 225
```

The returned `N` satisfies:
- Divisible by the stride factor `L` for the group
- Efficient for FFTW (smooth factorization)

---

## Utility Functions

### Plan Queries

```julia
subgrid_size(plan)     # (M₁, M₂, M₃) — dimensions of the subgrid
fullgrid_size(plan)    # (N₁, N₂, N₃) — dimensions of the full grid
stride_factors(plan)   # (L₁, L₂, L₃) — stride in each dimension
cfft_asu_size(plan)    # number of unique spectral coefficients
```

### Spectral Space Utilities

```julia
k2 = cfft_k2(plan, lattice)    # |k|² for each spectral ASU point
```

### Grid Conversions

Convert between the compact subgrid and the full periodic grid:

```julia
f_full = zeros(fullgrid_size(plan)...)
subgrid_to_fullgrid!(f_full, plan, f0)     # expand subgrid → full grid

f0_back = zeros(subgrid_size(plan)...)
fullgrid_to_subgrid!(f0_back, plan, f_full)  # extract full grid → subgrid
```

---

## Type Hierarchy

```
AbstractCFFTPlan
├── CFFTPlan{FP}        — forward-only
├── ICFFTPlan{BP}       — backward-only
└── AbstractCFFTPairPlan
    ├── GeneralCFFTPairPlan   — all 230 space groups
    └── CenteredCFFTPairPlan  — I/C/A/F centering optimization
```

All plan types support the same query functions (`subgrid_size`, `fullgrid_size`, etc.).
