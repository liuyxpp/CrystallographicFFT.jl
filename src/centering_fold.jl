# ============================================================================
# Centering fold on stride-2 subgrid
#
# Exploits the centering periodicity of the stride-2 subgrid f₀ to decompose
# a single M³ FFT into n_channels × (M/2)³ FFTs, achieving 2× (I/C) or 4× (F)
# FFT reduction on top of stride-2 decomposition.
#
# Mathematical basis:
#   f₀(n) = u(2n) has internal centering periodicity: f₀(n + c) = f₀(n)
#   where c = τ/2 (centering vector halved, in subgrid coords).
#
#   This allows alias fold: the M³-point DFT of f₀ can be decomposed into
#   n_ch DFTs of size H³ = (M/2)³, one per alive frequency class.
#
#   G₀(2k + off) = DFT_H[ g_off(n₀) · w_off(n₀) ](k)
#   where:
#     g_off(n₀) = Σ_{ε ∈ {0,1}³} (-1)^(off·ε) f₀(n₀ + H·ε)
#     w_off(n₀) = exp(-2πi off·n₀ / M)
# ============================================================================

using ..SymmetryOps: CenteringType, CentP, CentC, CentA, CentI, CentF

"""
    SubgridCenteringFoldPlan

Plan for alias-folding a stride-2 subgrid f₀ using its centering periodicity,
then FFT-ing each channel and assembling the non-zero entries of G₀.

# Fields
- `centering`: CenteringType (CentI, CentF, or CentC)
- `M`: subgrid dimensions (M₁, M₂, M₃)
- `H`: folded dimensions (M₁/2, M₂/2, M₃/2)
- `n_channels`: number of alive frequency classes
- `offsets`: parity offset per channel (each ∈ {0,1}³)
- `channel_bufs`: pre-allocated H³ complex buffers for folded data
- `channel_fft_plans`: FFTW plans for each channel
- `channel_fft_out`: pre-allocated H³ complex buffers for FFT output
- `twiddle_1d`: precomputed 1D twiddle tables, `twiddle_1d[c][d][n+1]`
- `sign_table`: precomputed 8-element sign arrays per channel
"""
struct SubgridCenteringFoldPlan
    centering::CenteringType
    M::NTuple{3,Int}
    H::NTuple{3,Int}
    n_channels::Int
    offsets::Vector{NTuple{3,Int}}
    channel_bufs::Vector{Array{ComplexF64,3}}
    channel_fft_plans::Vector{FFTW.cFFTWPlan{ComplexF64,-1,false,3,UnitRange{Int64}}}
    channel_ifft_plans::Vector{Any}  # ScaledPlan{ComplexF64, cFFTWPlan{...,1,...}, Float64}
    channel_fft_out::Vector{Array{ComplexF64,3}}
    twiddle_1d::Vector{NTuple{3,Vector{ComplexF64}}}  # [channel][dim][n+1]
    sign_table::Vector{NTuple{8,Int}}                  # [channel][(ez*4+ey*2+ex)+1]
end

"""
    plan_centering_fold(centering::CenteringType, M::NTuple{3,Int})

Create a plan for centering fold on a stride-2 subgrid of dimensions M.
"""
function plan_centering_fold(centering::CenteringType, M::NTuple{3,Int})
    @assert all(iseven, M) "Subgrid dimensions must be even for centering fold"
    H = M .÷ 2

    offsets = _alive_offsets(centering)
    n_ch = length(offsets)

    channel_bufs = [zeros(ComplexF64, H...) for _ in 1:n_ch]
    channel_fft_out = [zeros(ComplexF64, H...) for _ in 1:n_ch]
    channel_fft_plans = [plan_fft(channel_bufs[c], 1:3) for c in 1:n_ch]
    channel_ifft_plans = [plan_ifft(channel_fft_out[c], 1:3) for c in 1:n_ch]

    # P1: Precompute separable 1D twiddle tables
    # tw_d[n+1] = cispi(-2 * off_d * n / M_d) for n in 0:H_d-1
    twiddle_1d = Vector{NTuple{3,Vector{ComplexF64}}}(undef, n_ch)
    for c in 1:n_ch
        off = offsets[c]
        tw = ntuple(3) do d
            if off[d] == 0
                ones(ComplexF64, H[d])
            else
                [cispi(-2 * off[d] * n / M[d]) for n in 0:H[d]-1]
            end
        end
        twiddle_1d[c] = tw
    end

    # P5: Precompute sign table per channel
    # signs[(ez*4+ey*2+ex)+1] = (-1)^(off·ε)
    sign_table = Vector{NTuple{8,Int}}(undef, n_ch)
    for c in 1:n_ch
        off = offsets[c]
        signs = ntuple(8) do idx
            ex = (idx - 1) & 1
            ey = ((idx - 1) >> 1) & 1
            ez = ((idx - 1) >> 2) & 1
            1 - 2 * ((off[1]*ex + off[2]*ey + off[3]*ez) & 1)
        end
        sign_table[c] = signs
    end

    return SubgridCenteringFoldPlan(centering, M, H, n_ch, offsets,
        channel_bufs, channel_fft_plans, channel_ifft_plans, channel_fft_out,
        twiddle_1d, sign_table)
end

"""
    _alive_offsets(centering) → Vector{NTuple{3,Int}}

Return the parity offsets of alive (non-extinct) frequency classes.
"""
function _alive_offsets(centering::CenteringType)
    if centering == CentI
        # I-centering: h₁+h₂+h₃ even → 4 classes
        return [(0,0,0), (1,1,0), (1,0,1), (0,1,1)]
    elseif centering == CentF
        # F-centering: h all-same-parity → 2 classes
        return [(0,0,0), (1,1,1)]
    elseif centering == CentC
        # C-centering: h₁+h₂ even → 4 classes
        return [(0,0,0), (1,1,0), (0,0,1), (1,1,1)]
    elseif centering == CentA
        # A-centering: h₂+h₃ even → 4 classes
        return [(0,0,0), (0,1,1), (1,0,0), (1,1,1)]
    else
        error("Centering fold not applicable for $centering")
    end
end

"""
    centering_fold!(plan::SubgridCenteringFoldPlan, f0::AbstractArray{<:Real,3})

Fold the subgrid f₀ into n_channels alias-folded channels on H³.
Each channel c gets:
  g_off(n₀) = Σ_{ε∈{0,1}³} (-1)^(off·ε) · f₀(n₀ + H·ε) · twiddle(off, n₀)

For 4-channel centering (I/C/A), uses a fused single-pass Walsh-Hadamard
butterfly to compute all sign sums in 24 add/sub (vs 60 per-channel),
reading f₀ only once instead of 4 times.
"""
function centering_fold!(plan::SubgridCenteringFoldPlan,
                          f0::AbstractArray{<:Real,3})
    if plan.n_channels == 4
        _centering_fold_4ch!(plan, f0)
    else
        _centering_fold_2ch!(plan, f0)
    end
end

"""
2-channel kernel for F-centering: channel 0 (sum) + channel 1 (1,1,1) with twiddle.
"""
function _centering_fold_2ch!(plan::SubgridCenteringFoldPlan,
                               f0::AbstractArray{<:Real,3})
    H1, H2, H3 = plan.H
    buf0 = plan.channel_bufs[1]
    buf1 = plan.channel_bufs[2]
    tw1, tw2, tw3 = plan.twiddle_1d[2]
    signs = plan.sign_table[2]

    @inbounds for iz in 0:H3-1
        tw_z = tw3[iz+1]
        for iy in 0:H2-1
            tw_yz = tw2[iy+1] * tw_z
            for ix in 0:H1-1
                v000 = f0[ix+1,     iy+1,     iz+1]
                v100 = f0[ix+H1+1,  iy+1,     iz+1]
                v010 = f0[ix+1,     iy+H2+1,  iz+1]
                v110 = f0[ix+H1+1,  iy+H2+1,  iz+1]
                v001 = f0[ix+1,     iy+1,     iz+H3+1]
                v101 = f0[ix+H1+1,  iy+1,     iz+H3+1]
                v011 = f0[ix+1,     iy+H2+1,  iz+H3+1]
                v111 = f0[ix+H1+1,  iy+H2+1,  iz+H3+1]

                buf0[ix+1, iy+1, iz+1] = v000+v100+v010+v110+v001+v101+v011+v111

                val1 = (signs[1]*v000 + signs[2]*v100 +
                        signs[3]*v010 + signs[4]*v110 +
                        signs[5]*v001 + signs[6]*v101 +
                        signs[7]*v011 + signs[8]*v111)
                buf1[ix+1, iy+1, iz+1] = val1 * (tw1[ix+1] * tw_yz)
            end
        end
    end
end

"""
4-channel fused WHT kernel for I/C/A centering.

Reads 8 f₀ aliases ONCE per spatial point and computes all 8 Walsh-Hadamard
coefficients via a 3-stage butterfly (24 add/sub). Then selects the 4 alive
channels by precomputed WHT index mapping and applies separable twiddle.
"""
function _centering_fold_4ch!(plan::SubgridCenteringFoldPlan,
                               f0::AbstractArray{<:Real,3})
    H1, H2, H3 = plan.H
    buf1 = plan.channel_bufs[1]
    buf2 = plan.channel_bufs[2]
    buf3 = plan.channel_bufs[3]
    buf4 = plan.channel_bufs[4]

    # WHT index: offset (o1,o2,o3) → index o1+o2*2+o3*4+1 in the WHT tuple
    # (o1 varies fastest because stage-1 butterfly is over ε₁)
    off2 = plan.offsets[2]; idx2 = off2[1] + off2[2]*2 + off2[3]*4 + 1
    off3 = plan.offsets[3]; idx3 = off3[1] + off3[2]*2 + off3[3]*4 + 1
    off4 = plan.offsets[4]; idx4 = off4[1] + off4[2]*2 + off4[3]*4 + 1

    tw1_2, tw2_2, tw3_2 = plan.twiddle_1d[2]
    tw1_3, tw2_3, tw3_3 = plan.twiddle_1d[3]
    tw1_4, tw2_4, tw3_4 = plan.twiddle_1d[4]

    @inbounds for iz in 0:H3-1
        tz2 = tw3_2[iz+1]; tz3 = tw3_3[iz+1]; tz4 = tw3_4[iz+1]
        for iy in 0:H2-1
            tyz2 = tw2_2[iy+1] * tz2
            tyz3 = tw2_3[iy+1] * tz3
            tyz4 = tw2_4[iy+1] * tz4
            for ix in 0:H1-1
                # ── Read 8 f₀ aliases ONCE ──
                v000 = f0[ix+1,     iy+1,     iz+1]
                v100 = f0[ix+H1+1,  iy+1,     iz+1]
                v010 = f0[ix+1,     iy+H2+1,  iz+1]
                v110 = f0[ix+H1+1,  iy+H2+1,  iz+1]
                v001 = f0[ix+1,     iy+1,     iz+H3+1]
                v101 = f0[ix+H1+1,  iy+1,     iz+H3+1]
                v011 = f0[ix+1,     iy+H2+1,  iz+H3+1]
                v111 = f0[ix+H1+1,  iy+H2+1,  iz+H3+1]

                # ── 3-stage WHT butterfly (24 add/sub) ──
                # Stage 1: transform over ε₁
                a0_00 = v000 + v100;  a1_00 = v000 - v100
                a0_10 = v010 + v110;  a1_10 = v010 - v110
                a0_01 = v001 + v101;  a1_01 = v001 - v101
                a0_11 = v011 + v111;  a1_11 = v011 - v111
                # Stage 2: transform over ε₂
                b00_0 = a0_00 + a0_10;  b01_0 = a0_00 - a0_10
                b10_0 = a1_00 + a1_10;  b11_0 = a1_00 - a1_10
                b00_1 = a0_01 + a0_11;  b01_1 = a0_01 - a0_11
                b10_1 = a1_01 + a1_11;  b11_1 = a1_01 - a1_11
                # Stage 3: transform over ε₃ → WHT coefficients c_{o1,o2,o3}
                wht = (b00_0 + b00_1,   # c₀₀₀  idx=1
                       b10_0 + b10_1,   # c₁₀₀  idx=2
                       b01_0 + b01_1,   # c₀₁₀  idx=3
                       b11_0 + b11_1,   # c₁₁₀  idx=4
                       b00_0 - b00_1,   # c₀₀₁  idx=5
                       b10_0 - b10_1,   # c₁₀₁  idx=6
                       b01_0 - b01_1,   # c₀₁₁  idx=7
                       b11_0 - b11_1)   # c₁₁₁  idx=8

                # ── Write alive channels ──
                buf1[ix+1, iy+1, iz+1] = wht[1]  # (0,0,0): no twiddle
                buf2[ix+1, iy+1, iz+1] = wht[idx2] * (tw1_2[ix+1] * tyz2)
                buf3[ix+1, iy+1, iz+1] = wht[idx3] * (tw1_3[ix+1] * tyz3)
                buf4[ix+1, iy+1, iz+1] = wht[idx4] * (tw1_4[ix+1] * tyz4)
            end
        end
    end
end

"""
    fft_channels!(plan::SubgridCenteringFoldPlan)

Execute FFT on all folded channels (out-of-place: channel_bufs → channel_fft_out).
"""
function fft_channels!(plan::SubgridCenteringFoldPlan)
    @inbounds for c in 1:plan.n_channels
        mul!(plan.channel_fft_out[c], plan.channel_fft_plans[c], plan.channel_bufs[c])
    end
end

"""
    assemble_G0!(G0::AbstractArray{ComplexF64}, plan::SubgridCenteringFoldPlan)

Assemble G₀ from channel FFT outputs. Writes G₀(2k+off) for each alive class.
G₀ must be a flat Vector{ComplexF64} of length prod(M), reshaped as (M₁,M₂,M₃).
Non-alive entries are set to zero.
"""
function assemble_G0!(G0::AbstractArray{ComplexF64,3}, plan::SubgridCenteringFoldPlan)
    M1, M2, M3 = plan.M
    H1, H2, H3 = plan.H

    # Zero out everything first (extinct entries)
    fill!(G0, zero(ComplexF64))

    @inbounds for c in 1:plan.n_channels
        off = plan.offsets[c]
        fft_out = plan.channel_fft_out[c]

        for iz in 0:H3-1, iy in 0:H2-1, ix in 0:H1-1
            # Map: h = 2k + off, where k = (ix, iy, iz)
            h1 = 2*ix + off[1]
            h2 = 2*iy + off[2]
            h3 = 2*iz + off[3]
            G0[h1+1, h2+1, h3+1] = fft_out[ix+1, iy+1, iz+1]
        end
    end
end

# ============================================================================
# CenteredKRFFTPlan: compositional wrapper
# ============================================================================

"""
    CenteredKRFFTPlan

A composite plan that wraps a `GeneralCFFTPlan` with centering fold on the
stride-2 subgrid. Pipeline:

    u(N³) → pack_stride → f₀(M³) → centering_fold → n_ch × H³
          → FFT channels → assemble G₀(M³) → fast_reconstruct → spectral ASU
"""
struct CenteredKRFFTPlan
    krfft_plan::GeneralCFFTPlan          # inner KRFFT plan
    fold_plan::SubgridCenteringFoldPlan  # centering fold plan
    f0_buffer::Array{Float64,3}          # M³ real buffer for stride-2 subgrid
    G0_view::Array{ComplexF64,3}         # pre-stored reshape of work_buffer
end

"""
    plan_krfft_centered(spec_asu, ops_shifted; centering=:auto)

Create a centered KRFFT plan that combines stride-2 decomposition with
centering fold on the subgrid.

Falls back to plain `plan_krfft` if centering is P or L ≠ [2,2,2].
Returns a `CenteredKRFFTPlan` if centering fold is applicable, otherwise
a `GeneralCFFTPlan`.
"""
function plan_krfft_centered(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp};
                             centering::Union{CenteringType,Symbol}=:auto)
    N = spec_asu.N
    dim = length(N)

    # Auto-detect centering
    if centering === :auto
        cent = detect_centering_type(ops_shifted, NTuple{dim,Int}(N))
    else
        cent = centering::CenteringType
    end

    L = auto_L(ops_shifted)

    # Centering fold requires L=[2,2,2] and non-P centering
    if cent == CentP || !all(L .== 2) || dim != 3
        return plan_krfft(spec_asu, ops_shifted)
    end

    M_sub = NTuple{3,Int}(N[d] ÷ 2 for d in 1:3)

    # Check M is even (required for fold)
    if !all(iseven, M_sub)
        return plan_krfft(spec_asu, ops_shifted)
    end

    # Create inner KRFFT plan (operates on M³ subgrid)
    krfft_plan = plan_krfft(spec_asu, ops_shifted)

    # Create fold plan
    fold_plan = plan_centering_fold(cent, M_sub)

    # Allocate f₀ buffer
    f0_buffer = zeros(Float64, M_sub...)

    # P3: Pre-store G0 reshape view (avoids runtime Tuple(Vector{Int}) type instability)
    G0_view = reshape(krfft_plan.work_buffer, M_sub)

    return CenteredKRFFTPlan(krfft_plan, fold_plan, f0_buffer, G0_view)
end

"""
    pack_stride_real!(f0, u)

Extract stride-2 subgrid from real array u into real f₀ buffer.
f₀[n] = u[2n] for all n.
"""
function pack_stride_real!(f0::Array{Float64,3}, u::AbstractArray{<:Real,3})
    M1, M2, M3 = size(f0)
    @inbounds for k in 1:M3, j in 1:M2, i in 1:M1
        f0[i, j, k] = u[2*(i-1)+1, 2*(j-1)+1, 2*(k-1)+1]
    end
end

"""
    execute_centered_krfft!(plan::CenteredKRFFTPlan, u)

Full centered KRFFT pipeline:
1. pack_stride → f₀ (real M³)
2. centering_fold → n_ch channels on H³
3. FFT channels → channel spectra
4. assemble G₀ → work_buffer (M³)
5. fast_reconstruct → spectral ASU
"""
function execute_centered_krfft!(plan::CenteredKRFFTPlan,
                                  u::AbstractArray{<:Real,3})
    krfft = plan.krfft_plan
    fold = plan.fold_plan

    # 1. Extract stride-2 subgrid (real)
    pack_stride_real!(plan.f0_buffer, u)

    # 2. Centering fold: f₀(M³) → n_ch channels on H³
    centering_fold!(fold, plan.f0_buffer)

    # 3. FFT each channel
    fft_channels!(fold)

    # 4. Assemble G₀ into the KRFFT work_buffer (pre-stored view)
    assemble_G0!(plan.G0_view, fold)

    # 5. Reconstruct spectral ASU from G₀
    fast_reconstruct!(krfft)

    return krfft.output_buffer
end

"""
    fft_reconstruct_centered!(plan::CenteredKRFFTPlan)

Fast-path variant: assumes f₀ is already in plan.f0_buffer.
Executes centering fold → FFT channels → assemble G₀ → reconstruct.
"""
function fft_reconstruct_centered!(plan::CenteredKRFFTPlan)
    fold = plan.fold_plan
    krfft = plan.krfft_plan

    centering_fold!(fold, plan.f0_buffer)
    fft_channels!(fold)
    assemble_G0!(plan.G0_view, fold)
    fast_reconstruct!(krfft)

    return krfft.output_buffer
end

# ============================================================================
# Inverse Operations
# ============================================================================

"""
    ifft_channels!(plan::SubgridCenteringFoldPlan)

Execute IFFT on all folded channels (out-of-place: channel_fft_out → channel_bufs).
"""
function ifft_channels!(plan::SubgridCenteringFoldPlan)
    @inbounds for c in 1:plan.n_channels
        mul!(plan.channel_bufs[c], plan.channel_ifft_plans[c], plan.channel_fft_out[c])
    end
end

"""
    disassemble_G0!(plan::SubgridCenteringFoldPlan, G0::AbstractArray{ComplexF64,3})

Inverse of `assemble_G0!`: reads G₀(M³) and fills channel_fft_out arrays.
Extracts alive-parity entries from G₀ into the per-channel FFT output buffers.
"""
function disassemble_G0!(plan::SubgridCenteringFoldPlan,
                          G0::AbstractArray{ComplexF64,3})
    H1, H2, H3 = plan.H

    @inbounds for c in 1:plan.n_channels
        off = plan.offsets[c]
        fft_out = plan.channel_fft_out[c]

        for iz in 0:H3-1, iy in 0:H2-1, ix in 0:H1-1
            h1 = 2*ix + off[1]
            h2 = 2*iy + off[2]
            h3 = 2*iz + off[3]
            fft_out[ix+1, iy+1, iz+1] = G0[h1+1, h2+1, h3+1]
        end
    end
end

"""
    centering_unfold!(plan::SubgridCenteringFoldPlan, f0::AbstractArray{<:Real,3})

Inverse of `centering_fold!`: reconstruct f₀(M³) from n_ch channels on H³.

The inverse WHT is WHT/8 (self-inverse up to normalization). For alive channels,
extinct WHT coefficients are implicitly zero (centering symmetry guarantee).

    f₀(n₀ + Hε) = (1/8) Σ_off (-1)^(off·ε) · g_off(n₀) / twiddle(off, n₀)
"""
function centering_unfold!(plan::SubgridCenteringFoldPlan,
                            f0::AbstractArray{<:Real,3})
    if plan.n_channels == 4
        _centering_unfold_4ch!(plan, f0)
    elseif plan.n_channels == 2
        _centering_unfold_2ch!(plan, f0)
    end
end

"""
2-channel inverse kernel for F-centering.
Reconstructs f₀ from channel 0 (off=(0,0,0)) and channel 1 (off=(1,1,1)).
"""
function _centering_unfold_2ch!(plan::SubgridCenteringFoldPlan,
                                 f0::AbstractArray{<:Real,3})
    H1, H2, H3 = plan.H
    buf0 = plan.channel_bufs[1]  # g_{(0,0,0)}(n0)
    buf1 = plan.channel_bufs[2]  # g_{(1,1,1)}(n0) · twiddle
    tw1, tw2, tw3 = plan.twiddle_1d[2]
    signs = plan.sign_table[2]

    @inbounds for iz in 0:H3-1
        tw_z_conj = conj(tw3[iz+1])
        for iy in 0:H2-1
            tw_yz_conj = conj(tw2[iy+1]) * tw_z_conj
            for ix in 0:H1-1
                # Undo twiddle to get raw WHT coefficient
                tw_conj = conj(tw1[ix+1]) * tw_yz_conj
                g0 = real(buf0[ix+1, iy+1, iz+1])
                g1 = real(buf1[ix+1, iy+1, iz+1] * tw_conj)

                # Inverse WHT: f0(n0 + H*eps) = (1/8) sum_off (-1)^(off.eps) * g_off
                # Only 2 alive: off=(0,0,0) with g0, off=(1,1,1) with g1
                # All 6 extinct channels are zero
                for eps_idx in 1:8
                    s = signs[eps_idx]  # (-1)^((1,1,1)·eps)
                    val = (g0 + s * g1) / 8.0

                    ex = (eps_idx - 1) & 1
                    ey = ((eps_idx - 1) >> 1) & 1
                    ez = ((eps_idx - 1) >> 2) & 1
                    f0[ix+ex*H1+1, iy+ey*H2+1, iz+ez*H3+1] = val
                end
            end
        end
    end
end

"""
4-channel inverse kernel for I/C/A centering.
Reconstructs f₀ from 4 alive channels using inverse WHT.
"""
function _centering_unfold_4ch!(plan::SubgridCenteringFoldPlan,
                                 f0::AbstractArray{<:Real,3})
    H1, H2, H3 = plan.H

    # Build alive WHT indices (same mapping as forward)
    alive_wht_idx = [off[1] + off[2]*2 + off[3]*4 + 1 for off in plan.offsets]

    @inbounds for iz in 0:H3-1
        # Precompute twiddle conjugates for dim 3
        tw_z = [conj(plan.twiddle_1d[c][3][iz+1]) for c in 1:4]
        for iy in 0:H2-1
            # Precompute twiddle conjugates for dim 2*3
            tw_yz = [conj(plan.twiddle_1d[c][2][iy+1]) * tw_z[c] for c in 1:4]
            for ix in 0:H1-1
                # Get raw WHT coefficients by undoing twiddle
                # wht_coeff[wht_idx] = g_off(n0) / twiddle(off, n0)
                wht_coeffs = zeros(8)  # all 8 WHT slots, extinct = 0
                for c in 1:4
                    tw_full = conj(plan.twiddle_1d[c][1][ix+1]) * tw_yz[c]
                    wht_coeffs[alive_wht_idx[c]] = real(
                        plan.channel_bufs[c][ix+1, iy+1, iz+1] * tw_full)
                end

                # Inverse WHT (3-stage butterfly, same as forward since WHT = WHT^-1 up to scale)
                # Stage 1: dim 1 (stride=1)
                a0_00 = wht_coeffs[1] + wht_coeffs[2]; a1_00 = wht_coeffs[1] - wht_coeffs[2]
                a0_10 = wht_coeffs[3] + wht_coeffs[4]; a1_10 = wht_coeffs[3] - wht_coeffs[4]
                a0_01 = wht_coeffs[5] + wht_coeffs[6]; a1_01 = wht_coeffs[5] - wht_coeffs[6]
                a0_11 = wht_coeffs[7] + wht_coeffs[8]; a1_11 = wht_coeffs[7] - wht_coeffs[8]
                # Stage 2: dim 2 (stride=2)
                b00_0 = a0_00 + a0_10; b01_0 = a0_00 - a0_10
                b10_0 = a1_00 + a1_10; b11_0 = a1_00 - a1_10
                b00_1 = a0_01 + a0_11; b01_1 = a0_01 - a0_11
                b10_1 = a1_01 + a1_11; b11_1 = a1_01 - a1_11
                # Stage 3: dim 3 (stride=4) -> f0 values / 8
                inv8 = 1.0 / 8.0
                f0[ix+1,     iy+1,     iz+1]     = (b00_0 + b00_1) * inv8 # eps=(0,0,0)
                f0[ix+H1+1,  iy+1,     iz+1]     = (b10_0 + b10_1) * inv8 # eps=(1,0,0)
                f0[ix+1,     iy+H2+1,  iz+1]     = (b01_0 + b01_1) * inv8 # eps=(0,1,0)
                f0[ix+H1+1,  iy+H2+1,  iz+1]     = (b11_0 + b11_1) * inv8 # eps=(1,1,0)
                f0[ix+1,     iy+1,     iz+H3+1]  = (b00_0 - b00_1) * inv8 # eps=(0,0,1)
                f0[ix+H1+1,  iy+1,     iz+H3+1]  = (b10_0 - b10_1) * inv8 # eps=(1,0,1)
                f0[ix+1,     iy+H2+1,  iz+H3+1]  = (b01_0 - b01_1) * inv8 # eps=(0,1,0)
                f0[ix+H1+1,  iy+H2+1,  iz+H3+1]  = (b11_0 - b11_1) * inv8 # eps=(1,1,1)
            end
        end
    end
end

# ============================================================================
# Centered KRFFT Backward Transform
# ============================================================================

"""Compact inverse entry: maps orbit rep to weighted F̂ gather."""
struct InvCompactEntry
    h_idx::Int32           # Spectral index to gather from
    weight::ComplexF64     # Inverse weight A⁻¹[r, h]
end

"""Orbit expansion entry: maps G₀ position to its orbit representative."""
struct OrbitExpandEntry
    rep_idx::Int32         # Compact orbit rep index (1-based), 0 = dead
    phase::ComplexF64      # Phase factor conj(sym_phase)
end

"""
    CenteredKRFFTBackwardPlan

Backward plan for centered KRFFT: spectral ASU → f₀(M³).

Two paths:
- **Two-step** (fast, use_dense==false):
  Step 1: Compact gather F̂ → G₀_reps via block-diagonal A⁻¹.
  Step 2: Orbit expansion G₀_reps → full G₀ (sequential write).
  Then: disassemble_G₀ → IFFT → unfold.
- **Factored orbit** (use_dense==true):
  Step 1: c = B_sym_inv × F̂   (n_spec² ops)
  Step 2: f₀[m] = Re(c[orbit_id[m]]) / √|orbit|  (M³ lookups)
"""
struct CenteredKRFFTBackwardPlan
    use_dense::Bool

    # ── Two-step path ──
    inv_compact_table::Matrix{InvCompactEntry}  # (max_block_size, n_reps)
    block_sizes::Vector{Int32}                  # entries per rep
    orbit_expand::Vector{OrbitExpandEntry}       # (M_vol,) maps G₀→rep
    n_reps::Int
    G0_reps::Vector{ComplexF64}                 # workspace (n_reps,)

    # ── Factored orbit path (non-symmorphic groups) ──
    B_sym_inv::Matrix{ComplexF64}               # (n_spec, n_spec) inverse
    dense_orbit_id::Vector{Int32}               # (M_vol,) orbit index per position
    dense_orbit_norm::Vector{Float64}           # (n_orbits,) = 1/√|orbit|
    dense_coeffs::Vector{ComplexF64}            # (n_spec,) workspace

    # Shared state
    fold_plan::SubgridCenteringFoldPlan
    f0_buffer::Array{Float64, 3}
    G0_view::Array{ComplexF64, 3}
    subgrid_dims::NTuple{3,Int}
    n_spec::Int
end

"""
    plan_centered_ikrfft(spec_asu, ops_shifted, fwd_plan)

Create a centered KRFFT backward plan from the forward plan.

Algorithm (following M6 backward approach):
1. Extract representative ops (A8-like, one per parity class) and remaining
   point-group ops (G_rem, even-translation ops) from the full ops
2. Build orbits of G₀ positions under G_rem, mapping each position to its
   orbit representative with a phase factor
3. Build merged A matrix: A[h, r] = Σ_k a8_tw × sym_phase, where r is an
   orbit representative (compact index). This matrix is block-diagonal with
   square blocks (n_reps == n_spec)
4. Block-diagonal inversion of A → A⁻¹
5. Two-stage scatter fused into a single inv_recon_table:
   For each h, for each orbit rep r with A⁻¹[r,h] ≠ 0, expand r to its full
   orbit and scatter F̂[h] × A⁻¹[r,h] × conj(sym_phase) to all G₀ positions.
"""
function plan_centered_ikrfft(spec_asu::SpectralIndexing,
                               ops_shifted::Vector{SymOp},
                               fwd_plan::CenteredKRFFTPlan)
    krfft = fwd_plan.krfft_plan
    fold = fwd_plan.fold_plan
    N = Tuple(spec_asu.N)
    M = Tuple(krfft.subgrid_dims)
    dim = length(M)
    n_spec = length(spec_asu.points)
    n_ops = krfft.n_ops  # number of representative ops (parity classes)
    fwd_table = krfft.recon_table

    # ── 1. Extract remaining point-group ops (even translation) ──
    rem_ops = [op for op in ops_shifted if all(mod.(round.(Int, op.t), 2) .== 0)]

    # ── 2. Build set of ALL alive G₀ positions ──
    alive_parities = Set(Tuple(mod.(off, 2)) for off in fold.offsets)
    alive_set = Set{Int}()
    for k3 in 0:M[3]-1, k2 in 0:M[2]-1, k1 in 0:M[1]-1
        parity = (mod(k1, 2), mod(k2, 2), mod(k3, 2))
        if parity in alive_parities
            lin = 1 + k1 + M[1]*k2 + M[1]*M[2]*k3
            push!(alive_set, lin)
        end
    end

    # Collect positions referenced by spectral ASU (for seeding orbits)
    g0_pos_set = Set{Int}()
    for h in 1:n_spec, g in 1:n_ops
        push!(g0_pos_set, Int(fwd_table[g, h].buffer_idx))
    end

    # ── 3. Build orbits under G_rem ──
    g0_to_rep = Dict{Int, Int}()
    g0_to_phase = Dict{Int, ComplexF64}()

    for start_lin in g0_pos_set
        haskey(g0_to_rep, start_lin) && continue
        lin0 = start_lin - 1
        sq = [mod(lin0, M[1]), mod(div(lin0, M[1]), M[2]), div(lin0, M[1]*M[2])]

        orbit = Dict{Int, Tuple{Vector{Int}, ComplexF64}}()
        orbit[start_lin] = (sq, complex(1.0))
        worklist = [(sq, complex(1.0))]

        while !isempty(worklist)
            q_vec, q_phase = pop!(worklist)
            for op in rem_ops
                R = op.R
                t_half = round.(Int, op.t) .÷ 2
                rq = [mod(sum(Int(R[d2, d1]) * q_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]
                rlin = 1 + rq[1] + M[1]*rq[2] + M[1]*M[2]*rq[3]
                if !haskey(orbit, rlin) && rlin ∈ alive_set
                    sym_phase = cispi(-2 * sum(q_vec[d] * t_half[d] / M[d] for d in 1:dim))
                    new_phase = sym_phase * q_phase
                    orbit[rlin] = (rq, new_phase)
                    push!(worklist, (rq, new_phase))
                end
            end
        end

        rep_lin = minimum(keys(orbit))
        rep_phase = orbit[rep_lin][2]
        for (member_lin, (_, member_phase)) in orbit
            g0_to_rep[member_lin] = rep_lin
            g0_to_phase[member_lin] = member_phase / rep_phase
        end
    end

    # Build compact indexing for orbit representatives
    reps = sort(collect(Set(values(g0_to_rep))))
    rep_to_compact = Dict(r => i for (i, r) in enumerate(reps))
    n_reps = length(reps)

    # ── 4. Build per-h list of (compact_r, weight) entries ──
    # h_entries[h] = [(compact_r, weight), ...] — pre-indexed, no Dict scanning
    h_entries = [Vector{Tuple{Int, ComplexF64}}() for _ in 1:n_spec]
    for h in 1:n_spec
        # Accumulate A[h, compact_r] from forward table
        local_acc = Dict{Int, ComplexF64}()
        for g in 1:n_ops
            e = fwd_table[g, h]
            q_lin = Int(e.buffer_idx)
            rep_lin = g0_to_rep[q_lin]
            sym_phase = g0_to_phase[q_lin]
            compact_r = rep_to_compact[rep_lin]
            local_acc[compact_r] = get(local_acc, compact_r, zero(ComplexF64)) + e.weight * sym_phase
        end
        for (r, w) in local_acc
            push!(h_entries[h], (r, w))
        end
    end

    # Also build rep → list of specs (for union-find blocks)
    rep_to_specs = [Vector{Int}() for _ in 1:n_reps]
    for h in 1:n_spec
        for (r, _) in h_entries[h]
            push!(rep_to_specs[r], h)
        end
    end

    # ── 5. Union-find on spectral points sharing orbit reps ──
    parent2 = collect(1:n_spec)
    find2(x) = begin
        while parent2[x] != x
            parent2[x] = parent2[parent2[x]]
            x = parent2[x]
        end
        x
    end
    function unite2(a, b)
        a, b = find2(a), find2(b)
        a != b && (parent2[a] = b)
    end
    for r in 1:n_reps
        u_specs = unique(rep_to_specs[r])
        for i in 2:length(u_specs)
            unite2(u_specs[1], u_specs[i])
        end
    end

    block_map = Dict{Int, Vector{Int}}()
    for h in 1:n_spec
        root = find2(h)
        push!(get!(Vector{Int}, block_map, root), h)
    end

    # ── 6. Per-block inversion (array-indexed, O(Σ block²)) ──
    # inv_h_entries[h] = [(compact_r, inv_weight), ...]
    inv_h_entries = [Vector{Tuple{Int, ComplexF64}}() for _ in 1:n_spec]

    for (_, h_idxs) in block_map
        sort!(h_idxs)

        # Collect the set of reps used by this block
        rep_set = Set{Int}()
        for h in h_idxs
            for (r, _) in h_entries[h]
                push!(rep_set, r)
            end
        end
        rep_list = sort(collect(rep_set))

        n_h = length(h_idxs)
        n_r = length(rep_list)
        if n_h != n_r
            return _plan_centered_ikrfft_dense(spec_asu, fwd_plan, rem_ops)
        end

        rep_to_local = Dict(r => i for (i, r) in enumerate(rep_list))
        A_sub = zeros(ComplexF64, n_h, n_r)
        for (local_h, h) in enumerate(h_idxs)
            for (r, w) in h_entries[h]
                A_sub[local_h, rep_to_local[r]] += w
            end
        end

        A_inv = inv(A_sub)

        for (local_r, compact_r) in enumerate(rep_list)
            for (local_h, h) in enumerate(h_idxs)
                w = A_inv[local_r, local_h]
                abs(w) < 1e-15 && continue
                push!(inv_h_entries[h], (compact_r, w))
            end
        end
    end

    # ── 7. Fuse into single scatter table (array-indexed) ──
    # Build orbit membership: compact_r → [(q_lin, conj_phase), ...]
    rep_members = [Vector{Tuple{Int, ComplexF64}}() for _ in 1:n_reps]
    for (q_lin, rep_lin) in g0_to_rep
        compact_r = rep_to_compact[rep_lin]
        push!(rep_members[compact_r], (q_lin, conj(g0_to_phase[q_lin])))
    end

    # ── Build two-step tables ──

    # Step 1 table: inv_compact_table[g, r] = InvCompactEntry(h_idx, weight)
    # For each rep r, gather from the h values in its block with A⁻¹ weights.
    # inv_h_entries[h] = [(compact_r, inv_w), ...] → transpose to per-rep.
    rep_inv_entries = [Vector{Tuple{Int, ComplexF64}}() for _ in 1:n_reps]
    for h in 1:n_spec
        for (r, w) in inv_h_entries[h]
            push!(rep_inv_entries[r], (h, w))
        end
    end

    max_block = maximum(length(rep_inv_entries[r]) for r in 1:n_reps)
    inv_compact_table = Matrix{InvCompactEntry}(undef, max_block, n_reps)
    block_sizes_vec = Vector{Int32}(undef, n_reps)
    for r in 1:n_reps
        entries = rep_inv_entries[r]
        block_sizes_vec[r] = Int32(length(entries))
        for (g, (h, w)) in enumerate(entries)
            inv_compact_table[g, r] = InvCompactEntry(Int32(h), w)
        end
        # Fill remaining slots (unused but must be initialized)
        for g in (length(entries)+1):max_block
            inv_compact_table[g, r] = InvCompactEntry(Int32(1), zero(ComplexF64))
        end
    end

    # Step 2 table: orbit_expand[q_lin] = OrbitExpandEntry(rep_idx, phase)
    M_vol = prod(M)
    orbit_expand_vec = Vector{OrbitExpandEntry}(undef, M_vol)
    for q in 1:M_vol
        orbit_expand_vec[q] = OrbitExpandEntry(Int32(0), zero(ComplexF64))
    end
    for (q_lin, rep_lin) in g0_to_rep
        compact_r = rep_to_compact[rep_lin]
        orbit_expand_vec[q_lin] = OrbitExpandEntry(Int32(compact_r),
                                                   conj(g0_to_phase[q_lin]))
    end

    G0_view = reshape(krfft.work_buffer, M)
    empty_dense = Matrix{ComplexF64}(undef, 0, 0)
    G0_reps_buf = Vector{ComplexF64}(undef, n_reps)

    return CenteredKRFFTBackwardPlan(
        false,  # use_dense
        inv_compact_table, block_sizes_vec, orbit_expand_vec, n_reps, G0_reps_buf,
        empty_dense, Int32[], Float64[], ComplexF64[],
        fold, fwd_plan.f0_buffer, G0_view,
        M, n_spec)
end

"""
    execute_centered_ikrfft!(bplan, F_spec, f0_out)

Centered KRFFT backward: F̂[n_spec] → f₀[M³].

Dispatches to orbit-scatter path or dense-matrix path based on plan.
"""
function execute_centered_ikrfft!(bplan::CenteredKRFFTBackwardPlan,
                                   F_spec::AbstractVector{ComplexF64},
                                   f0_out::AbstractArray{Float64, 3})
    if bplan.use_dense
        # Factored orbit path: c = B_sym_inv × F̂, then orbit expand to f₀
        _factored_orbit_inv!(f0_out, F_spec, bplan.B_sym_inv,
                             bplan.dense_coeffs, bplan.dense_orbit_id,
                             bplan.dense_orbit_norm)
        return f0_out
    end

    # Two-step path
    fold = bplan.fold_plan
    G0 = bplan.G0_view

    # Step 1: Compact gather F̂ → G₀_reps (block-diagonal A⁻¹)
    _inv_recon_compact!(bplan.G0_reps, F_spec,
                        bplan.inv_compact_table, bplan.block_sizes, bplan.n_reps)

    # Step 2: Orbit expand G₀_reps → full G₀ (sequential write)
    _orbit_expand!(G0, bplan.G0_reps, bplan.orbit_expand)

    # Step 3: Disassemble G₀ → channel FFT outputs
    disassemble_G0!(fold, G0)

    # Step 4: IFFT channels
    ifft_channels!(fold)

    # Step 5: Centering unfold → f₀
    centering_unfold!(fold, f0_out)

    return f0_out
end

"""
    ifft_unrecon_centered!(bplan, F_spec)

SCFT fast path: F̂ → f₀. Result written to bplan.f0_buffer.
Symmetric counterpart of `fft_reconstruct_centered!`.
"""
function ifft_unrecon_centered!(bplan::CenteredKRFFTBackwardPlan,
                                 F_spec::AbstractVector{ComplexF64})
    execute_centered_ikrfft!(bplan, F_spec, bplan.f0_buffer)
end

"""Step 1: Compact gather F̂ → G₀_reps via block-diagonal A⁻¹."""
function _inv_recon_compact!(G0_reps::Vector{ComplexF64},
                              F_spec::AbstractVector{ComplexF64},
                              table::Matrix{InvCompactEntry},
                              block_sizes::Vector{Int32},
                              n_reps::Int)
    @inbounds for r in 1:n_reps
        val = zero(ComplexF64)
        for g in 1:block_sizes[r]
            e = table[g, r]
            val += e.weight * F_spec[e.h_idx]
        end
        G0_reps[r] = val
    end
end

"""Step 2: Orbit expand G₀_reps → full G₀ (sequential write by linear index)."""
function _orbit_expand!(G0::AbstractArray{ComplexF64,3},
                         G0_reps::Vector{ComplexF64},
                         orbit_expand::Vector{OrbitExpandEntry})
    G0_flat = vec(G0)
    @inbounds for q in eachindex(orbit_expand)
        e = orbit_expand[q]
        if e.rep_idx > 0
            G0_flat[q] = e.phase * G0_reps[e.rep_idx]
        else
            G0_flat[q] = zero(ComplexF64)
        end
    end
end

"""Factored orbit inverse: c = B_sym_inv × F̂, then f₀[m] = Re(c[orbit_id[m]]) × norm."""
function _factored_orbit_inv!(f0_out::AbstractArray{Float64, 3},
                               F_spec::AbstractVector{ComplexF64},
                               B_sym_inv::Matrix{ComplexF64},
                               coeffs::Vector{ComplexF64},
                               orbit_id::Vector{Int32},
                               orbit_norm::Vector{Float64})
    # Step 1: c = B_sym_inv × F̂  (n_spec² ops)
    mul!(coeffs, B_sym_inv, F_spec)

    # Step 2: f₀[m] = Re(c[orbit_id[m]]) × orbit_norm[orbit_id[m]]
    f0_flat = vec(f0_out)
    @inbounds for m in eachindex(f0_flat)
        k = orbit_id[m]
        if k > 0
            f0_flat[m] = real(coeffs[k]) * orbit_norm[k]
        else
            f0_flat[m] = 0.0
        end
    end
end

"""
    _plan_centered_ikrfft_dense(spec_asu, fwd_plan, rem_ops)

Dense symmetric-subspace fallback for groups where orbit blocks are non-square
(e.g., Ia-3d SG 230 with non-symmorphic ops).

Algorithm (scalable version using orbit enumeration):
1. Enumerate orbits of M-grid points under G_M = {(R, t÷2) : (R,t) ∈ rem_ops}
2. Build V_sym sparse-like: each orbit gives one column (normalized indicator)
3. Build pipeline matrix B restricted to orbit representatives (n_sym columns)
4. B_sym = B_restricted (square, n_spec × n_sym)
5. Precompute F̂→f₀: dense_inv = V_sym × B_sym⁻¹ (M³ × n_spec)
"""
function _plan_centered_ikrfft_dense(spec_asu::SpectralIndexing,
                                      fwd_plan::CenteredKRFFTPlan,
                                      rem_ops::Vector{SymOp})
    krfft = fwd_plan.krfft_plan
    fold = fwd_plan.fold_plan
    M = Tuple(krfft.subgrid_dims)
    dim = length(M)
    M_vol = prod(M)
    n_spec = length(spec_asu.points)

    # ── 1. Enumerate M-grid orbits under G_M ──
    G_M_ops = [(round.(Int, op.R), round.(Int, op.t) .÷ 2) for op in rem_ops]

    orbit_id = zeros(Int, M_vol)  # orbit_id[m] = which orbit does m belong to
    orbits = Vector{Vector{Int}}()  # orbits[k] = list of M-grid flat indices

    for m_flat in 1:M_vol
        orbit_id[m_flat] != 0 && continue  # already assigned

        # BFS to find the full orbit of m_flat
        orbit = [m_flat]
        orbit_set = Set{Int}([m_flat])
        queue = [m_flat]
        while !isempty(queue)
            cur = popfirst!(queue)
            cur0 = cur - 1
            mv = [mod(cur0, M[1]), mod(div(cur0, M[1]), M[2]),
                  div(cur0, M[1]*M[2])]
            for (R_m, s_m) in G_M_ops
                gm = [mod(sum(R_m[d, :] .* mv) + s_m[d], M[d]) for d in 1:dim]
                gm_flat = 1 + gm[1] + M[1]*gm[2] + M[1]*M[2]*gm[3]
                if gm_flat ∉ orbit_set
                    push!(orbit_set, gm_flat)
                    push!(orbit, gm_flat)
                    push!(queue, gm_flat)
                end
            end
        end

        push!(orbits, orbit)
        k = length(orbits)
        for idx in orbit
            orbit_id[idx] = k
        end
    end

    n_sym = length(orbits)
    @assert n_sym == n_spec (
        "Symmetric subspace dim ($n_sym) ≠ n_spec ($n_spec). " *
        "Cannot build dense inverse for this group.")

    # ── 2. Build V_sym via orbit-representative approach ──
    # V_sym[m, k] = 1/√|orbit_k|  if m ∈ orbit_k, else 0
    # Instead of dense V_sym (M_vol × n_sym), we build the product V_sym × X
    # directly. But for the dense_inv = V_sym × B_sym⁻¹, we need V_sym.
    # For moderate M_vol this is feasible as a dense matrix.
    V_sym = zeros(Float64, M_vol, n_sym)
    for (k, orbit) in enumerate(orbits)
        norm_k = 1.0 / sqrt(length(orbit))
        for idx in orbit
            V_sym[idx, k] = norm_k
        end
    end

    # ── 3. Build B_sym directly (pipeline applied to symmetrized unit vectors) ──
    # B_sym[h, k] = Σ_{m ∈ orbit_k} (1/√|orbit_k|) × B[h, m]
    # We compute this by running the forward pipeline on each orbit's
    # representative vector (uniform over orbit members).
    G0_view = reshape(krfft.work_buffer, M)
    B_sym = zeros(ComplexF64, n_spec, n_sym)
    for (k, orbit) in enumerate(orbits)
        fill!(fwd_plan.f0_buffer, 0.0)
        norm_k = 1.0 / sqrt(length(orbit))
        for idx in orbit
            fwd_plan.f0_buffer[idx] = norm_k
        end
        centering_fold!(fold, fwd_plan.f0_buffer)
        fft_channels!(fold)
        assemble_G0!(G0_view, fold)
        fast_reconstruct!(krfft)
        B_sym[:, k] = krfft.output_buffer
    end

    # ── 4. Build factored inverse: B_sym_inv + orbit tables ──
    B_sym_inv = ComplexF64.(inv(B_sym))  # (n_spec × n_spec)

    # Build orbit_id and orbit_norm vectors for f₀-level expansion
    orbit_id_vec = Vector{Int32}(undef, M_vol)
    for m in 1:M_vol
        orbit_id_vec[m] = Int32(orbit_id[m])
    end
    orbit_norm_vec = [1.0 / sqrt(length(orbit)) for orbit in orbits]
    coeffs_buf = Vector{ComplexF64}(undef, n_spec)

    # Build return plan with empty two-step tables
    empty_compact = Matrix{InvCompactEntry}(undef, 0, 0)
    empty_blocks = Int32[]
    empty_expand = OrbitExpandEntry[]

    return CenteredKRFFTBackwardPlan(
        true,   # use_dense
        empty_compact, empty_blocks, empty_expand, 0, ComplexF64[],
        B_sym_inv, orbit_id_vec, orbit_norm_vec, coeffs_buf,
        fold, fwd_plan.f0_buffer, G0_view,
        M, n_spec)
end


