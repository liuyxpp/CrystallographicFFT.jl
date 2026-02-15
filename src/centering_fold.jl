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
function plan_krfft_centered(spec_asu::SpectralIndexing, ops_shifted::Vector{<:SymOp};
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

"""
    CenteredKRFFTBackwardPlan

Backward plan for centered KRFFT: spectral ASU → f₀(M³).

Uses M2-style per-fiber B⁻¹ inversion (always d×d square) to recover G₀,
then centering unfold chain (disassemble → IFFT channels → centering unfold).

Optimized with:
- **CSR compact table**: skips zero-weight entries, pre-applies conjugation
- **Orbit reduction**: computes inv_recon at orbit reps only, expands via
  G₀(R^T q) = e^{2πi q·s/M} × G₀(q)
"""
struct CenteredKRFFTBackwardPlan
    # CSR compact inv_recon: nnz entries for orbit reps only
    inv_offsets::Vector{Int32}       # (n_orbits + 1,) CSR row pointers
    inv_spec_idx::Vector{Int32}      # (nnz,) spectral indices
    inv_weight::Vector{ComplexF64}   # (nnz,) weights (conj pre-applied)
    n_orbits::Int

    # Orbit expand: rep → all M_vol positions
    orbit_rep_pos::Vector{Int32}     # (n_orbits,) linear index of each rep
    orbit_member_oid::Vector{Int32}  # (M_vol,) orbit index for each position
    orbit_member_phase::Vector{ComplexF64}  # (M_vol,) phase factor per position

    # Workspace for orbit-level values
    G0_reps::Vector{ComplexF64}      # (n_orbits,) inv_recon output per rep

    # Centering unfold chain
    fold_plan::SubgridCenteringFoldPlan
    f0_buffer::Array{Float64, 3}
    G0_view::Array{ComplexF64, 3}
    subgrid_dims::NTuple{3,Int}
    n_spec::Int
end

"""
    plan_centered_ikrfft(spec_asu, ops_shifted, fwd_plan)

Create an optimized centered KRFFT backward plan.

Builds CSR compact inv_recon table (zero entries removed, conj pre-applied)
and spatial orbit structure for orbit-based reduction.
"""
function plan_centered_ikrfft(spec_asu::SpectralIndexing,
                               ops_shifted::Vector{<:SymOp},
                               fwd_plan::CenteredKRFFTPlan)
    krfft = fwd_plan.krfft_plan
    fold = fwd_plan.fold_plan
    M = Tuple(krfft.subgrid_dims)
    dim = length(M)
    M_vol = prod(M)
    n_spec = length(spec_asu.points)

    # ── 1. Build M2-style inv_recon_table ──
    m2bwd = plan_m2_backward(spec_asu, ops_shifted)
    d = m2bwd.d
    table = m2bwd.inv_recon_table

    # ── 2. Build spatial orbits under G_rem (even-translation ops) ──
    rem_ops = [(round.(Int, op.R), round.(Int, op.t) .÷ 2)
               for op in ops_shifted
               if all(mod.(round.(Int, op.t), 2) .== 0)]

    orbit_id = zeros(Int32, M_vol)    # position → orbit index
    orbit_phase = zeros(ComplexF64, M_vol)  # position → phase factor
    orbits_rep = Int32[]              # orbit index → representative position

    for m in 1:M_vol
        orbit_id[m] != 0 && continue
        # New orbit: m is the representative
        push!(orbits_rep, Int32(m))
        oid = Int32(length(orbits_rep))
        orbit_id[m] = oid
        orbit_phase[m] = complex(1.0)

        # BFS to find all orbit members with phase tracking
        # Phase relation: G₀(R^T q mod M) = e^{2πi q·s/M} × G₀(q)
        queue = [m]
        while !isempty(queue)
            cur = popfirst!(queue)
            cur0 = cur - 1
            qv = (mod(cur0, M[1]), mod(div(cur0, M[1]), M[2]),
                  div(cur0, M[1] * M[2]))
            cur_phase = orbit_phase[cur]

            for (R, s) in rem_ops
                # Apply R^T to frequency q
                rq = ntuple(d1 -> mod(sum(R[d2, d1] * qv[d2] for d2 in 1:dim),
                                      M[d1]), Val(3))
                rq_lin = 1 + rq[1] + M[1] * rq[2] + M[1] * M[2] * rq[3]

                if orbit_id[rq_lin] == 0
                    # Phase: G₀(R^T q) = e^{2πi q·s/M} × G₀(q)
                    ph = cispi(2 * sum(qv[dd] * s[dd] / M[dd] for dd in 1:dim))
                    orbit_id[rq_lin] = oid
                    orbit_phase[rq_lin] = ph * cur_phase
                    push!(queue, rq_lin)
                end
            end
        end
    end

    n_orbits = length(orbits_rep)

    # ── 3. Build CSR compact table for orbit reps only ──
    # Count non-zero entries per rep
    offsets = Vector{Int32}(undef, n_orbits + 1)
    offsets[1] = Int32(1)
    for i in 1:n_orbits
        q = orbits_rep[i]
        cnt = Int32(0)
        for a in 1:d
            abs(table[a, q].weight) > 1e-15 && (cnt += 1)
        end
        offsets[i + 1] = offsets[i] + cnt
    end

    nnz = Int(offsets[n_orbits + 1] - 1)
    spec_idx = Vector{Int32}(undef, nnz)
    weight = Vector{ComplexF64}(undef, nnz)

    for i in 1:n_orbits
        q = orbits_rep[i]
        j = Int(offsets[i])
        for a in 1:d
            e = table[a, q]
            abs(e.weight) > 1e-15 || continue
            weight[j] = e.weight
            # Use negative spec_idx to signal conjugation at runtime
            spec_idx[j] = e.conj_flag ? -e.spec_idx : e.spec_idx
            j += 1
        end
    end


    G0_view = reshape(krfft.work_buffer, M)
    G0_reps = Vector{ComplexF64}(undef, n_orbits)

    return CenteredKRFFTBackwardPlan(
        offsets, spec_idx, weight, n_orbits,
        orbits_rep, orbit_id, orbit_phase,
        G0_reps,
        fold, fwd_plan.f0_buffer, G0_view,
        M, n_spec)
end

"""
    execute_centered_ikrfft!(bplan, F_spec, f0_out)

Centered KRFFT backward: F̂[n_spec] → f₀[M³].

Steps:
1. CSR inv_recon at orbit reps only: F̂ → G₀_reps
2. Orbit expand: G₀_reps → G₀ (all M_vol positions)
3. Disassemble G₀ → channel FFT outputs
4. IFFT channels
5. Centering unfold → f₀
"""
function execute_centered_ikrfft!(bplan::CenteredKRFFTBackwardPlan,
                                   F_spec::AbstractVector{ComplexF64},
                                   f0_out::AbstractArray{Float64, 3})
    G0 = bplan.G0_view

    # Step 1+2: Optimized inv_recon with orbit reduction
    _inv_recon_orbit!(G0, F_spec, bplan)

    # Step 3: Disassemble G₀ → channel FFT outputs
    fold = bplan.fold_plan
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

"""
Optimized inv_recon: CSR gather at orbit reps + phase-expand to M_vol.

Uses negative spec_idx to signal conjugation (branchless sign trick).
"""
function _inv_recon_orbit!(G0::AbstractArray{ComplexF64, 3},
                            F_spec::AbstractVector{ComplexF64},
                            bplan::CenteredKRFFTBackwardPlan)
    offsets = bplan.inv_offsets
    sidx = bplan.inv_spec_idx
    wt = bplan.inv_weight
    reps = bplan.orbit_rep_pos
    G0_reps = bplan.G0_reps
    n_orbits = bplan.n_orbits

    # Step 1: CSR gather at orbit reps only
    @inbounds for i in 1:n_orbits
        val = zero(ComplexF64)
        for j in offsets[i]:(offsets[i + 1] - Int32(1))
            h = sidx[j]
            if h > 0
                val += wt[j] * F_spec[h]
            else
                val += wt[j] * conj(F_spec[-h])
            end
        end
        G0_reps[i] = val
    end

    # Step 2: Orbit expand to all M_vol positions
    G0_flat = vec(G0)
    oid = bplan.orbit_member_oid
    oph = bplan.orbit_member_phase
    @inbounds for q in eachindex(G0_flat)
        G0_flat[q] = oph[q] * G0_reps[oid[q]]
    end
end

# ============================================================================
# Centered SCFT Diffusion Plan (fwd + spectral K + bwd)
# ============================================================================

"""
    CenteredSCFTPlan

SCFT diffusion plan using forward + spectral K multiply + backward.

Computes `f₀_new = IKRFFT(K(h) · KRFFT(f₀))` where:
- `KRFFT` is the M7 forward transform (centering fold + FFT + recon)
- `K(h) = exp(-Δs · |k(h)|²)` is the diffusion kernel
- `IKRFFT` is the M7 backward transform (CSR+orbit inv_recon + unfold)

For high-symmetry groups (|G| ≥ 32), this is 1.3–3× faster than the Q-matrix
approach because spectral K multiply on n_spec ASU points replaces Q-multiply
on M_vol fibers.

# Fields
- `fwd_plan`: Forward M7 KRFFT plan
- `bwd_plan`: Backward M7 KRFFT plan (CSR + orbit optimized)
- `K_spec`: Pre-computed diffusion kernel for each spectral ASU point
- `F_spec`: Workspace for spectral coefficients
- `n_spec`: Number of spectral ASU points
"""
struct CenteredSCFTPlan
    fwd_plan::CenteredKRFFTPlan
    bwd_plan::CenteredKRFFTBackwardPlan
    K_spec::Vector{Float64}
    F_spec::Vector{ComplexF64}
    n_spec::Int
end

"""
    plan_centered_scft(spec_asu, ops_shifted, N, Δs, lattice) -> CenteredSCFTPlan

Create an SCFT diffusion plan using the fwd+bwd route.

# Arguments
- `spec_asu`: Spectral ASU from `calc_spectral_asu`
- `ops_shifted`: Shifted symmetry operations
- `N`: Full grid dimensions, e.g. `(64, 64, 64)`
- `Δs`: Chain contour step size
- `lattice`: Lattice vectors as columns of a matrix (a₁|a₂|a₃)

# Notes
Since `K(Rᵀh) = K(h)` for all symmetry operations R (the diffusion kernel
depends only on |k|² which is invariant under point group operations), the
spectral K multiply is exact on the ASU—no approximation is involved.

When `Δs` changes (e.g., variable step-size SCFT), only `K_spec` needs to be
recomputed (O(n_spec)), not the entire plan.
"""
function plan_centered_scft(spec_asu::SpectralIndexing,
                             ops_shifted::Vector{<:SymOp},
                             N::Tuple,
                             Δs::Float64,
                             lattice::AbstractMatrix)
    # Build forward and backward plans
    fwd_plan = plan_krfft_centered(spec_asu, ops_shifted)
    bwd_plan = plan_centered_ikrfft(spec_asu, ops_shifted, fwd_plan)

    n_spec = length(spec_asu.points)

    # Pre-compute diffusion kernel K(h) = exp(-Δs · |k(h)|²)
    # Reciprocal lattice vectors: k = 2π B⁻ᵀ h
    recip_B = 2π * inv(Matrix(lattice))'
    D = length(N)

    K_spec = Vector{Float64}(undef, n_spec)
    for (i, pt) in enumerate(spec_asu.points)
        # Centered frequency vector
        h_centered = [pt.idx[d] >= N[d] ÷ 2 ? pt.idx[d] - N[d] : pt.idx[d]
                      for d in 1:D]
        k_vec = recip_B * h_centered
        K_spec[i] = exp(-dot(k_vec, k_vec) * Δs)
    end

    F_spec = Vector{ComplexF64}(undef, n_spec)

    return CenteredSCFTPlan(fwd_plan, bwd_plan, K_spec, F_spec, n_spec)
end

"""
    execute_centered_scft!(plan::CenteredSCFTPlan, f0::Array{Float64,3})

Apply the SCFT diffusion operator to stride-L subgrid data `f₀` in-place.

Hot path: `f₀ → KRFFT → K·F̂ → IKRFFT → f₀'`

This replaces the M7+Q pipeline's Q-multiply step with a simple spectral
scalar multiply, which is much faster for high-symmetry groups.
"""
function execute_centered_scft!(plan::CenteredSCFTPlan,
                                 f0::Array{Float64,3})
    fwd = plan.fwd_plan
    bwd = plan.bwd_plan
    K = plan.K_spec
    F = plan.F_spec
    n_spec = plan.n_spec

    # Step 1: Forward M7 KRFFT — f₀ → F̂[n_spec]
    # f₀ is already the stride-L subgrid, copy directly into buffer
    copyto!(fwd.f0_buffer, f0)
    F_out = fft_reconstruct_centered!(fwd)
    @inbounds for i in 1:n_spec
        F[i] = F_out[i]
    end

    # Step 2: Spectral K multiply — F̂(h) *= K(h)
    @inbounds @simd for i in 1:n_spec
        F[i] *= K[i]
    end

    # Step 3: Backward M7 IKRFFT — F̂ → f₀'
    execute_centered_ikrfft!(bwd, F, f0)
end

"""
    update_kernel!(plan::CenteredSCFTPlan, spec_asu, N, Δs_new, lattice)

Update the diffusion kernel for a new `Δs` value without rebuilding the plan.
This is O(n_spec) — much faster than rebuilding Q matrices for M7+Q.
"""
function update_kernel!(plan::CenteredSCFTPlan,
                         spec_asu::SpectralIndexing,
                         N::Tuple,
                         Δs_new::Float64,
                         lattice::AbstractMatrix)
    recip_B = 2π * inv(Matrix(lattice))'
    D = length(N)
    K = plan.K_spec
    for (i, pt) in enumerate(spec_asu.points)
        h_centered = [pt.idx[d] >= N[d] ÷ 2 ? pt.idx[d] - N[d] : pt.idx[d]
                      for d in 1:D]
        k_vec = recip_B * h_centered
        K[i] = exp(-dot(k_vec, k_vec) * Δs_new)
    end
end
