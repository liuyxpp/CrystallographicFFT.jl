# ============================================================================
# Plan struct definitions for Device-Agnostic Crystallographic FFT
# ============================================================================
#
# All structs are fully parametric — no concrete types like ComplexF64 or
# FFTW.cFFTWPlan appear.  Type parameters are inferred from buffer/plan
# arguments at construction time.
#
# T ∈ {Float32, Float64} throughout.
#
# This file is `include`d into the CrystallographicFFT module scope.
# KernelAbstractions and AbstractFFTs are loaded by planning.jl / kernels.jl.
# ============================================================================

# ── Forward Plan (General M2) ────────────────────────────────────────────────

"""
    ForwardPlan{T, D, ...}

Device-agnostic forward KRFFT plan.  Stores FFT sub-plan, SoA reconstruction
table, and all working buffers on the target device.

# Type parameters
- `T<:AbstractFloat` — element type (`Float64` or `Float32`)
- `D`                — spatial dimension (2 or 3)
- `P`                — FFT plan type (FFTW or cuFFT, inferred)
- `VA`               — complex vector type (`Vector{Complex{T}}` or `CuVector`)
- `AA`               — D-dim complex array type
- `VI`               — index vector type (`Vector{Int32}` or `CuVector{Int32}`)
"""
struct ForwardPlan{T<:AbstractFloat, D,
                   P,
                   VA<:AbstractVector{Complex{T}},
                   AA<:AbstractArray{Complex{T}, D},
                   VI<:AbstractVector{Int32}} <: AbstractFFTs.Plan{T}
    # ── FFT ──
    fft_plan::P
    input_buffer::VA          # flat complex input  (M_vol,)
    work_buffer::VA           # flat complex output (M_vol,)
    output_buffer::VA         # spectral ASU output (n_spec,)

    # ── Precomputed views (reshape of flat buffers, zero-alloc) ──
    input_view::AA            # reshape(input_buffer, M)
    work_view::AA             # reshape(work_buffer, M)

    # ── SoA reconstruct table (execute-time, on device) ──
    recon_fiber_idx::VI       # (n_ops × n_spec,) → work_buffer linear index
    recon_weight::VA          # (n_ops × n_spec,) → combined phase weight
    n_ops::Int                # = prod(L) = fiber length
    n_spec::Int               # spectral ASU size

    # ── Pmmm separable fast path ──
    is_pmmm::Bool
    phase_factors::Vector{VA} # 1D separable phase factors per dimension (length D)

    # ── Metadata ──
    M::NTuple{D, Int}         # subgrid dims
    N::NTuple{D, Int}         # full grid dims
    L::NTuple{D, Int}         # stride factors
end

# ── Backward Plan (General M2) ───────────────────────────────────────────────

"""
    BackwardPlan{T, D, ...}

Device-agnostic backward KRFFT plan (inverse of `ForwardPlan`).
"""
struct BackwardPlan{T<:AbstractFloat, D,
                    IP,
                    VA<:AbstractVector{Complex{T}},
                    AA<:AbstractArray{Complex{T}, D},
                    VI<:AbstractVector{Int32}} <: AbstractFFTs.Plan{T}
    # ── IFFT ──
    ifft_plan::IP
    Y_buffer::VA              # inv_recon output / IFFT input (M_vol,)
    f0_buffer::VA             # IFFT output (M_vol,)
    Y_view::AA                # reshape(Y_buffer, M)
    f0_view::AA               # reshape(f0_buffer, M)

    # ── SoA inv_recon table ──
    inv_work_idx::VI          # (d × M_vol,) → F_work index
    inv_weight::VA            # (d × M_vol,) → combined weight
    F_work::VA                # (2n_spec,) workspace: [F; conj(F)]
    d::Int                    # fiber length = prod(L)
    n_spec::Int

    # ── Pmmm separable fast path ──
    is_separable::Bool
    inv_phase_factors::Vector{VA}  # 1D inverse twiddle factors (length D)

    # ── Metadata ──
    M::NTuple{D, Int}
    N::NTuple{D, Int}
    L::NTuple{D, Int}
end

# ── Centering Fold Plan ──────────────────────────────────────────────────────

"""
    CenteringFoldPlan{T, ...}

Plan for centering fold / unfold on stride-2 subgrids.
Supports 2-channel (F centering) and 4-channel (I/C/A centering).

On GPU, uses batched CUFFT via 4D arrays (H1×H2×H3×n_ch) for a single
kernel launch instead of n_ch sequential FFT calls.
"""
struct CenteringFoldPlan{T<:AbstractFloat, P, IP, BP, BIP,
                         VA<:AbstractArray{Complex{T}, 3},
                         BA<:AbstractArray{Complex{T}, 4},
                         VT<:AbstractVector{Complex{T}},
                         VI<:AbstractVector{Int32}}
    centering::Symbol             # :P, :I, :F, :C, :A
    M::NTuple{3, Int}         # subgrid dims (must be even)
    H::NTuple{3, Int}         # folded dims = M .÷ 2
    n_channels::Int
    offsets::Vector{NTuple{3, Int}}  # (n_ch,) — small, stays on CPU

    # ── Per-channel buffers & FFT plans (CPU path) ──
    channel_bufs::Vector{VA}
    channel_fft_plans::Vector{P}
    channel_ifft_plans::Vector{IP}
    channel_fft_out::Vector{VA}

    # ── Batched buffers & FFT plans (GPU path: single CUFFT call) ──
    batch_buf::BA              # (H1, H2, H3, n_ch)
    batch_fft_out::BA          # (H1, H2, H3, n_ch)
    batch_fft_plan::BP         # batched plan_fft over dims (1,2,3)
    batch_ifft_plan::BIP       # batched plan_ifft over dims (1,2,3)

    # ── Twiddle factors (per-channel, per-dimension, on device) ──
    twiddle_1d::Vector{NTuple{3, VT}}
    sign_table::Vector{NTuple{8, Int}}  # small constants, stays CPU

    # ── GPU fused assemble: parity → channel index (0=dead) ──
    alive_mask::VI                      # (8,) on device for GPU, Vector{Int32} for CPU
end

# ── Centered Forward Plan (composition) ──────────────────────────────────────

"""
    CenteredForwardPlan{T, ...}

Wraps a `ForwardPlan` + `CenteringFoldPlan` for centered lattices.
Pipeline: f₀(M³) → centering_fold → n_ch × H³ → FFT → assemble G₀ → reconstruct
"""
struct CenteredForwardPlan{T<:AbstractFloat,
                           FP<:ForwardPlan{T},
                           SP<:CenteringFoldPlan{T},
                           AR<:AbstractArray{T, 3},
                           AC<:AbstractArray{Complex{T}, 3}}
    krfft_plan::FP
    fold_plan::SP
    f0_buffer::AR             # M³ real buffer
    G0_view::AC               # reshaped view of krfft_plan.work_buffer → (M...)
end

# ── Centered Backward Plan ───────────────────────────────────────────────────

"""
    CenteredBackwardPlan{T, ...}

Backward plan for centered lattices.  Uses CSR orbit-based inverse
reconstruction + centering unfold.
"""
struct CenteredBackwardPlan{T<:AbstractFloat,
                            VA<:AbstractVector{Complex{T}},
                            VI<:AbstractVector{Int32},
                            SP<:CenteringFoldPlan{T},
                            AR<:AbstractArray{T, 3},
                            AC<:AbstractArray{Complex{T}, 3}}
    # ── CSR orbit inv_recon ──
    inv_offsets::VI           # (n_orbits + 1,) CSR row pointers
    inv_spec_idx::VI          # (nnz,) spectral indices
    inv_weight::VA            # (nnz,) combined weights
    n_orbits::Int
    orbit_rep_pos::VI         # (n_orbits,) linear position of orbit rep in M-grid
    orbit_member_oid::VI      # (M_vol,) → orbit ID for each grid point
    orbit_member_phase::VA    # (M_vol,) → phase factor for orbit expansion
    G0_reps::VA               # (n_orbits,) workspace for orbit-rep values

    # ── Centering unfold chain ──
    fold_plan::SP
    f0_buffer::AR
    G0_view::AC
    M::NTuple{3, Int}
    n_spec::Int
end
