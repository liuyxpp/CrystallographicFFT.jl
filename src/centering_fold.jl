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
        channel_bufs, channel_fft_plans, channel_fft_out,
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
