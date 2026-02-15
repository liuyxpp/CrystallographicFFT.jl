# ============================================================================
# Centering Pre-fold: KRFFT III decomposition for centered lattices
#
# Implements block-partition pre-folding that decomposes a centered-lattice
# field on N³ into k channels on (N/2)³, each capturing a parity class of
# the full spectrum. Each channel is then processed via direct FFT.
# ============================================================================

using ..SymmetryOps: CenteringType, CentP, CentC, CentA, CentI, CentF
using ..SymmetryOps: detect_centering_type, strip_centering, get_ops
using ..SpectralIndexing: SpectralIndexing as SpectralASU
using ..SpectralIndexing: calc_spectral_asu

export CenteringPreFoldPlan, CenteredKRFFTPlan
export plan_krfft_centered, execute_centered_krfft!

# ============================================================================
# Pre-fold Plan: channel definitions
# ============================================================================

"""
    CenteringPreFoldPlan

Pre-computed plan for centering decomposition.
Defines channels (parity classes) and their relationship to the full spectrum.
"""
struct CenteringPreFoldPlan
    centering::CenteringType
    N_full::NTuple{3,Int}
    N_sub::NTuple{3,Int}
    n_channels::Int
    parities::Vector{NTuple{3,Int}}
    shift_offsets::Vector{NTuple{3,Int}}
end

function plan_centering_prefold(centering::CenteringType, N::Tuple)
    N_full = NTuple{3,Int}(N)
    Nh = N_full .÷ 2

    if centering == CentI
        return CenteringPreFoldPlan(centering, N_full, Nh, 4,
            [(0,0,0), (1,1,0), (1,0,1), (0,1,1)],
            [(Nh[1],0,0), (0,Nh[2],0), (0,0,Nh[3])])
    elseif centering == CentF
        return CenteringPreFoldPlan(centering, N_full, Nh, 2,
            [(0,0,0), (1,1,1)], [(Nh[1],0,0)])
    elseif centering == CentC
        N_sub = (Nh[1], Nh[2], N_full[3])
        return CenteringPreFoldPlan(centering, N_full, N_sub, 2,
            [(0,0,0), (1,1,0)], [(Nh[1],0,0)])
    elseif centering == CentA
        N_sub = (N_full[1], Nh[2], Nh[3])
        return CenteringPreFoldPlan(centering, N_full, N_sub, 2,
            [(0,0,0), (0,1,1)], [(0,Nh[2],0)])
    else
        error("CentP has no pre-fold")
    end
end

# ============================================================================
# Pre-fold execution
# ============================================================================

function centering_prefold!(channels::Vector{Array{ComplexF64,3}},
                            u::Array{<:Real,3},
                            plan::CenteringPreFoldPlan)
    if plan.centering == CentI
        _prefold_I!(channels, u, plan)
    elseif plan.centering == CentF
        _prefold_F!(channels, u, plan)
    elseif plan.centering == CentC
        _prefold_C!(channels, u, plan)
    elseif plan.centering == CentA
        _prefold_A!(channels, u, plan)
    end
end

function _prefold_I!(channels::Vector{Array{ComplexF64,3}},
                     u::Array{<:Real,3}, plan::CenteringPreFoldPlan)
    Nf = plan.N_full
    Nh = plan.N_sub
    signs = ((1,1,1,1), (1,-1,-1,1), (1,-1,1,-1), (1,1,-1,-1))
    @inbounds for iz in 0:Nh[3]-1, iy in 0:Nh[2]-1, ix in 0:Nh[1]-1
        v0 = u[ix+1, iy+1, iz+1]
        vx = u[ix+Nh[1]+1, iy+1, iz+1]
        vy = u[ix+1, iy+Nh[2]+1, iz+1]
        vz = u[ix+1, iy+1, iz+Nh[3]+1]
        terms = (v0, vx, vy, vz)
        for c in 1:4
            s = signs[c]; p = plan.parities[c]
            tw = cispi(-2 * (p[1]*ix/Nf[1] + p[2]*iy/Nf[2] + p[3]*iz/Nf[3]))
            channels[c][ix+1,iy+1,iz+1] = 2.0*tw*(s[1]*terms[1]+s[2]*terms[2]+s[3]*terms[3]+s[4]*terms[4])
        end
    end
end

function _prefold_F!(channels::Vector{Array{ComplexF64,3}},
                     u::Array{<:Real,3}, plan::CenteringPreFoldPlan)
    Nf = plan.N_full; Nh = plan.N_sub
    @inbounds for iz in 0:Nh[3]-1, iy in 0:Nh[2]-1, ix in 0:Nh[1]-1
        v0 = u[ix+1, iy+1, iz+1]
        vs = u[ix+Nh[1]+1, iy+1, iz+1]
        channels[1][ix+1,iy+1,iz+1] = 4.0 * (v0 + vs)
        tw = cispi(-2 * (ix/Nf[1] + iy/Nf[2] + iz/Nf[3]))
        channels[2][ix+1,iy+1,iz+1] = 4.0 * (v0 - vs) * tw
    end
end

function _prefold_C!(channels::Vector{Array{ComplexF64,3}},
                     u::Array{<:Real,3}, plan::CenteringPreFoldPlan)
    Nf = plan.N_full; Ns = plan.N_sub; Hx = Nf[1]÷2
    @inbounds for iz in 0:Ns[3]-1, iy in 0:Ns[2]-1, ix in 0:Ns[1]-1
        v0 = u[ix+1, iy+1, iz+1]
        vs = u[ix+Hx+1, iy+1, iz+1]
        channels[1][ix+1,iy+1,iz+1] = 2.0 * (v0 + vs)
        tw = cispi(-2 * (ix/Nf[1] + iy/Nf[2]))
        channels[2][ix+1,iy+1,iz+1] = 2.0 * (v0 - vs) * tw
    end
end

function _prefold_A!(channels::Vector{Array{ComplexF64,3}},
                     u::Array{<:Real,3}, plan::CenteringPreFoldPlan)
    Nf = plan.N_full; Ns = plan.N_sub; Hy = Nf[2]÷2
    @inbounds for iz in 0:Ns[3]-1, iy in 0:Ns[2]-1, ix in 0:Ns[1]-1
        v0 = u[ix+1, iy+1, iz+1]
        vs = u[ix+1, iy+Hy+1, iz+1]
        channels[1][ix+1,iy+1,iz+1] = 2.0 * (v0 + vs)
        tw = cispi(-2 * (iy/Nf[2] + iz/Nf[3]))
        channels[2][ix+1,iy+1,iz+1] = 2.0 * (v0 - vs) * tw
    end
end

# ============================================================================
# Per-channel plan: direct FFT only
# ============================================================================

struct ChannelPlan
    fft_plan::Any
    fft_input::Array{ComplexF64,3}
    fft_output::Array{ComplexF64,3}
end

function _plan_channel(N_sub::Tuple)
    inp = zeros(ComplexF64, N_sub)
    out = zeros(ComplexF64, N_sub)
    return ChannelPlan(plan_fft(inp), inp, out)
end

function _execute_channel!(ch::ChannelPlan)
    mul!(ch.fft_output, ch.fft_plan, ch.fft_input)
end

# ============================================================================
# CenteredKRFFTPlan
# ============================================================================

struct ChannelMergeEntry
    channel::Int32
    fft_idx::Int32
end

struct CenteredKRFFTPlan
    prefold::CenteringPreFoldPlan
    channel_plans::Vector{ChannelPlan}
    channel_buffers::Vector{Array{ComplexF64,3}}
    full_spec_asu::SpectralASU
    merge_table::Vector{ChannelMergeEntry}
    output_buffer::Vector{ComplexF64}
    centering::CenteringType
    N_full::NTuple{3,Int}
    N_sub::NTuple{3,Int}
    n_channels::Int
end

function plan_krfft_centered(ops::Vector{<:SymOp}, N::Tuple)
    centering = detect_centering_type(ops, N)
    centering == CentP && error("No centering detected. Use plan_krfft instead.")
    return _build_centered_plan(ops, N, centering)
end

function plan_krfft_centered(sg_num::Int, N::Tuple; dim::Int=3)
    ops = get_ops(sg_num, dim, N)
    return plan_krfft_centered(ops, N)
end

function _build_centered_plan(ops::Vector{<:SymOp}, N::Tuple, centering::CenteringType)
    prefold = plan_centering_prefold(centering, N)
    N_sub = prefold.N_sub
    full_spec = calc_spectral_asu(ops, 3, N)

    n_ch = prefold.n_channels
    channel_plans = [_plan_channel(N_sub) for _ in 1:n_ch]
    channel_buffers = [zeros(ComplexF64, N_sub...) for _ in 1:n_ch]
    merge_table = _build_merge_table(full_spec, prefold, N, N_sub)

    n_unmapped = count(e -> e.channel == 0, merge_table)
    @info "CenteredKRFFTPlan: centering=$centering, N=$N→$(N_sub), " *
          "channels=$n_ch, full_spec=$(length(full_spec.points)), unmapped=$n_unmapped"

    return CenteredKRFFTPlan(
        prefold, channel_plans, channel_buffers,
        full_spec, merge_table, zeros(ComplexF64, length(full_spec.points)),
        centering, NTuple{3,Int}(N), N_sub, n_ch)
end

"""Map each full-grid spectral ASU point directly to (channel, FFT_linear_index)."""
function _build_merge_table(full_spec::SpectralASU, prefold::CenteringPreFoldPlan,
                            N_full::Tuple, N_sub::Tuple)
    n_full = length(full_spec.points)
    merge = Vector{ChannelMergeEntry}(undef, n_full)
    parities = prefold.parities

    for (i, pt) in enumerate(full_spec.points)
        h = pt.idx
        # Parity: only for halved dimensions, 0 for non-halved
        par = ntuple(d -> N_sub[d] < N_full[d] ? mod(h[d], 2) : 0, 3)
        ch_idx = findfirst(==(par), parities)
        if ch_idx === nothing
            merge[i] = ChannelMergeEntry(Int32(0), Int32(0))
            continue
        end
        # Map to sub-grid frequency
        h1 = N_sub[1] < N_full[1] ? mod((h[1]-par[1])÷2, N_sub[1]) : mod(h[1], N_sub[1])
        h2 = N_sub[2] < N_full[2] ? mod((h[2]-par[2])÷2, N_sub[2]) : mod(h[2], N_sub[2])
        h3 = N_sub[3] < N_full[3] ? mod((h[3]-par[3])÷2, N_sub[3]) : mod(h[3], N_sub[3])
        fft_idx = 1 + h1 + N_sub[1]*h2 + N_sub[1]*N_sub[2]*h3
        merge[i] = ChannelMergeEntry(Int32(ch_idx), Int32(fft_idx))
    end
    return merge
end

# ============================================================================
# Execution
# ============================================================================

function execute_centered_krfft!(plan::CenteredKRFFTPlan, u::Array{<:Real,3})
    centering_prefold!(plan.channel_buffers, u, plan.prefold)
    for (c, ch) in enumerate(plan.channel_plans)
        ch.fft_input .= plan.channel_buffers[c]
        _execute_channel!(ch)
    end
    _merge_channels!(plan)
    return plan.output_buffer
end

function _merge_channels!(plan::CenteredKRFFTPlan)
    out = plan.output_buffer
    mt = plan.merge_table
    @inbounds for i in 1:length(out)
        e = mt[i]
        if e.channel == 0
            out[i] = zero(ComplexF64)
        else
            out[i] = vec(plan.channel_plans[e.channel].fft_output)[e.fft_idx]
        end
    end
end
