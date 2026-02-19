# [Real-Valued CFFT](@id real-valued-cfft)

The `rcfft!` / `ircfft!` API exploits the Hermitian symmetry of real-valued fields to halve the FFT computation, providing **1.5–2× forward** and **2–3× backward** speedup over the standard `cfft!` / `icfft!` API.

## Motivation

In SCFT and most physical simulations, the real-space field $f(\mathbf{r})$ is always real-valued. The standard `cfft!` uses a complex-to-complex FFT (`fft`), which computes both positive and negative frequencies — redundant for real inputs since $F(-\mathbf{h}) = \overline{F(\mathbf{h})}$.

The `rcfft!` API uses `rfft` / `irfft` instead:
- **`rfft`**: real input $M^3$ → half-spectrum output $\hat{M}_1 \times M_2 \times M_3$, where $\hat{M}_1 = M_1 \div 2 + 1$
- **`irfft`**: half-spectrum input → real output $M^3$

## API

```julia
using CrystallographicFFT

N = (64, 64, 64)
sg = 221  # Pm-3m

# Bidirectional plan
rpair = plan_rcfft_pair(N, sg, 3)

# Forward: real subgrid → spectral ASU
f0 = rand(subgrid_size(rpair)...)
F̂ = Vector{ComplexF64}(undef, cfft_asu_size(rpair))
rcfft!(F̂, rpair, f0)

# Backward: spectral ASU → real subgrid
ircfft!(f0, rpair, F̂)
```

### Single-Direction Plans

```julia
fwd = plan_rcfft(N, sg, 3)            # → RCFFTPlan
rcfft!(F̂, fwd, f0)

bwd = plan_ircfft(fwd)                # → IRCFFTPlan
ircfft!(f0, bwd, F̂)
```

### Compatibility

All query functions work identically:

```julia
subgrid_size(rpair)     # same as cfft pair
fullgrid_size(rpair)    # same
cfft_asu_size(rpair)    # same spectral ASU
cfft_k2(rpair, lattice) # same wave vectors
make_diffusion_kernel(rpair, Δs, lattice)  # same kernel
```

## How It Works

### Signed Index Encoding

The reconstruction table uses **signed indices** to handle the half-spectrum boundary:

- `recon_fiber_idx[k] > 0`: access `rfft` output directly at index `i`
- `recon_fiber_idx[k] < 0`: access `conj(rfft_output[|i|])` (Hermitian conjugate)

This avoids constructing the full complex spectrum or copying data.

### Forward Path

```
f₀(real M³) → rfft → Ĝ₀(M̂₁ × M₂ × M₃) → signed-index reconstruct → F̂(n_spec)
```

The forward reconstruction reads from the half-spectrum, using signed indices to transparently handle conjugate access.

### Backward Path

```
F̂(n_spec) → signed-index inv_reconstruct → Ŷ₀(M̂₁ × M₂ × M₃) → irfft → f₀(real M³)
```

The backward reconstruction fills **only the half-spectrum** ($\hat{M}_1 \times M_2 \times M_3$ instead of $M^3$), halving the scatter work. The `irfft` directly outputs real data — no `complex → real` copy needed.

## Scope

- Works for **all 230 space groups**, including centered lattice groups (F/I/C/A)
- Internally always uses the **general path** (no centered-specific optimization)
- For centered groups, `rcfft` general+rfft achieves comparable or better performance than `cfft` centered+complex-fft

## CPU Benchmarks

### Forward: `rcfft!` vs `cfft!`

| Group | |G| | N=32 | N=64 | N=128 |
|-------|-----|------|------|-------|
| Pm-3m | 48 | 1.58× | 1.63× | **1.84×** |
| Pmmm | 8 | 1.52× | 1.47× | 1.15× |
| P4/mmm | 16 | 1.15× | 1.26× | **1.62×** |
| P2/m | 4 | 1.10× | 1.38× | **1.97×** |
| P-1 | 2 | 1.44× | 1.62× | **1.83×** |

### Backward: `ircfft!` vs `icfft!`

| Group | |G| | N=32 | N=64 | N=128 |
|-------|-----|------|------|-------|
| Pm-3m | 48 | **1.96×** | **1.90×** | **2.23×** |
| Pmmm | 8 | **2.19×** | **2.65×** | **2.69×** |
| P4/mmm | 16 | **2.28×** | **2.23×** | **2.55×** |
| P2/m | 4 | **2.56×** | **2.85×** | **3.59×** |
| P-1 | 2 | **3.00×** | **2.47×** | **2.46×** |

Backward speedup is consistently higher (2–3.6×) because:
1. `irfft` is ~2× faster than `ifft`
2. Inverse reconstruction fills only the half-spectrum (work halved)
3. No `[F; conj(F)]` buffer needed
4. Output is directly real — no complex→real copy

### Centered Groups (CPU, N=128)

| Group | |G| | cfft μs | rcfft μs | fwd× | icfft μs | ircfft μs | bwd× |
|-------|-----|---------|----------|------|----------|-----------|------|
| Fm-3m | 48 | 2761 | 1457 | **1.90×** | 7321 | 3291 | **2.22×** |
| Im-3m | 48 | 3248 | 1679 | **1.93×** | 7820 | 3517 | **2.22×** |

## GPU Benchmarks (N=128, RTX 2080 Ti)

### General Groups

| Group | |G| | GPU rcfft μs | GPU cfft μs | rcfft/cfft | GPU ircfft μs | GPU icfft μs | ircfft/icfft |
|-------|------|-------------|-------------|------------|--------------|-------------|--------------|
| Pm-3m | 48 | 175.9 | 521.1 | **2.96×** | 265.9 | 456.9 | **1.72×** |
| Pmmm | 8 | 471.1 | 475.0 | 1.01× | 314.8 | 523.4 | **1.66×** |
| P4/mmm | 16 | 280.1 | 322.1 | 1.15× | 289.4 | 485.1 | **1.68×** |
| P-1 | 2 | 756.6 | 1033.1 | **1.37×** | 678.1 | 1071.8 | **1.58×** |

### Centered Groups

| Group | |G| | GPU rcfft μs | GPU cfft μs | rcfft/cfft | GPU ircfft μs | GPU icfft μs | ircfft/icfft |
|-------|------|-------------|-------------|------------|--------------|-------------|--------------|
| Fm-3m | 192 | 152.1 | 190.8 | **1.25×** | 256.5 | 436.2 | **1.70×** |
| Im-3m | 96 | 157.9 | 198.7 | **1.26×** | 260.7 | 437.7 | **1.68×** |
| Fd-3m | 192 | 152.1 | 191.9 | **1.26×** | 255.8 | 441.3 | **1.72×** |
| I4/mmm | 32 | 215.2 | 256.7 | **1.19×** | 284.4 | 455.1 | **1.60×** |

GPU `rcfft!` requires no extra code — `plan_rcfft_pair` accepts `array_type=CuArray{Float64}` and automatically uses CUFFT `plan_rfft`.

## Accuracy

All transforms achieve machine-precision accuracy:

| Metric | Value |
|--------|-------|
| `rcfft!` ↔ `ircfft!` roundtrip | < 1e-10 |
| `rcfft!` vs `cfft!` cross-validation | < 1e-12 |
| `rcfft!` vs full-grid FFT | < 1e-12 |
| SCFT diffusion vs full-grid | < 1e-10 |
