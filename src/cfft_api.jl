# ============================================================================
# Public CFFT API — wraps internal KRFFT forward/backward plans
# ============================================================================

module CFFTApi

using LinearAlgebra
using ..SymmetryOps: SymOp, get_ops, detect_centering_type, CentP
using ..ASU: find_optimal_shift
using ..SpectralIndexing: calc_spectral_asu, SpectralIndexing, get_k_vector
using ..KRFFT: auto_L, GeneralForwardPlan, GeneralBackwardPlan,
    CenteredForwardPlan, CenteredBackwardPlan,
    plan_krfft, plan_m2_backward, fft_reconstruct!, execute_m2_backward!,
    plan_krfft_centered, plan_centered_ikrfft,
    fft_reconstruct_centered!, execute_centered_ikrfft!,
    CenteredSCFTPlan
using ..QFusedKRFFT: _build_fill_map

# ---- Pair plans (bidirectional) ----
export AbstractCFFTPlan, AbstractCFFTPairPlan
export GeneralCFFTPairPlan, CenteredCFFTPairPlan
export plan_cfft_pair

# ---- Single-direction plans ----
export CFFTPlan, ICFFTPlan
export plan_cfft, plan_icfft

# ---- Transforms ----
export cfft!, icfft!

# ---- Utilities ----
export make_diffusion_kernel, update_diffusion_kernel!
export cfft_k2
export subgrid_to_fullgrid!, fullgrid_to_subgrid!
export subgrid_size, fullgrid_size, stride_factors, cfft_asu_size

# ============================================================================
# Type hierarchy
# ============================================================================

"""
    AbstractCFFTPlan

Abstract base type for all Crystallographic FFT plans.
All plan types share metadata fields: `spec_asu`, `n_spec`, `sg_num`, `dim`,
`N`, `M`, `L`, `fill_map`.
"""
abstract type AbstractCFFTPlan end

"""
    AbstractCFFTPairPlan <: AbstractCFFTPlan

Abstract type for bidirectional CFFT plans (contain both forward and backward).
"""
abstract type AbstractCFFTPairPlan <: AbstractCFFTPlan end

# ============================================================================
# PairPlans — bidirectional (fwd + bwd)
# ============================================================================

"""
    GeneralCFFTPairPlan <: AbstractCFFTPairPlan

Bidirectional CFFT plan for all 230 space groups (General/M2 path).
Supports both `cfft!` and `icfft!`.
"""
struct GeneralCFFTPairPlan{FP<:GeneralForwardPlan, BP<:GeneralBackwardPlan} <: AbstractCFFTPairPlan
    fwd::FP
    bwd::BP
    spec_asu::SpectralIndexing
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

"""
    CenteredCFFTPairPlan <: AbstractCFFTPairPlan

Bidirectional CFFT plan for I/C/A/F-centered lattices (centering fold path).
Supports both `cfft!` and `icfft!`.
"""
struct CenteredCFFTPairPlan{FP<:CenteredForwardPlan, BP<:CenteredBackwardPlan} <: AbstractCFFTPairPlan
    fwd::FP
    bwd::BP
    spec_asu::SpectralIndexing
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

# ============================================================================
# Single-direction plans
# ============================================================================

"""
    CFFTPlan{FP} <: AbstractCFFTPlan

Forward-only Crystallographic FFT plan. Use with `cfft!`.

The type parameter `FP` hides the General/Centered implementation detail.
"""
struct CFFTPlan{FP, SO<:SymOp} <: AbstractCFFTPlan
    fwd::FP
    ops_shifted::Vector{SO}   # stored for plan_icfft(fwd)
    spec_asu::SpectralIndexing
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

"""
    ICFFTPlan{BP} <: AbstractCFFTPlan

Backward-only (inverse) Crystallographic FFT plan. Use with `icfft!`.

The type parameter `BP` hides the General/Centered implementation detail.
"""
struct ICFFTPlan{BP} <: AbstractCFFTPlan
    bwd::BP
    spec_asu::SpectralIndexing
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

# ============================================================================
# Shared plan geometry computation
# ============================================================================

"""Internal: compute plan geometry (ops, spec_asu, L, M, fill_map, use_centered)."""
function _plan_geometry(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                         method::Symbol=:auto) where D
    @assert D == dim "N must have $dim elements"

    direct_ops = get_ops(sg_num, dim, N)
    _, shifted_ops = find_optimal_shift(direct_ops, N)

    spec_asu = calc_spectral_asu(shifted_ops, dim, N)
    n_spec = length(spec_asu.points)

    L_vec = auto_L(shifted_ops)
    L = NTuple{3,Int}(L_vec)
    M = NTuple{3,Int}(N[d] ÷ L[d] for d in 1:3)

    use_centered = false
    if method == :centered || method == :auto
        centering = detect_centering_type(shifted_ops, N)
        if centering != CentP && all(L_vec .== 2) && dim == 3 && all(iseven, M)
            use_centered = true
        end
    end
    if method == :general
        use_centered = false
    end

    fill_map = _build_fill_map(shifted_ops, L_vec, collect(M), collect(N), D)

    return shifted_ops, spec_asu, n_spec, L, M, fill_map, use_centered
end

# ============================================================================
# plan_cfft — forward-only
# ============================================================================

"""
    plan_cfft(N, sg_num, dim; method=:auto) → CFFTPlan

Construct a forward-only Crystallographic FFT plan.

# Arguments
- `N::NTuple`: Full grid dimensions, e.g. `(64,64,64)`
- `sg_num::Int`: Space group number (1–230)
- `dim::Int`: Spatial dimension (2 or 3)
- `method::Symbol`: `:auto` (default), `:general`, `:centered`
"""
function plan_cfft(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                    method::Symbol=:auto) where D
    ops_s, spec_asu, n_spec, L, M, fill_map, use_centered =
        _plan_geometry(N, sg_num, dim; method)

    if use_centered
        fwd = plan_krfft_centered(spec_asu, ops_s)
    else
        fwd = plan_krfft(spec_asu, ops_s)
    end
    return CFFTPlan(fwd, ops_s, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
end

# ============================================================================
# plan_icfft — backward-only
# ============================================================================

"""
    plan_icfft(fwd::CFFTPlan) → ICFFTPlan

Construct a backward plan from an existing forward plan (efficient, shares geometry).
"""
function plan_icfft(fwd_plan::CFFTPlan)
    fwd = fwd_plan.fwd
    ops_s = fwd_plan.ops_shifted
    spec_asu = fwd_plan.spec_asu
    if fwd isa CenteredForwardPlan
        bwd = plan_centered_ikrfft(spec_asu, ops_s, fwd)
    elseif fwd isa GeneralForwardPlan
        bwd = plan_m2_backward(spec_asu, ops_s)
    else
        error("Unknown forward plan type: $(typeof(fwd))")
    end
    return ICFFTPlan(bwd, spec_asu, fwd_plan.n_spec,
                     fwd_plan.sg_num, fwd_plan.dim,
                     fwd_plan.N, fwd_plan.M, fwd_plan.L, fwd_plan.fill_map)
end

"""
    plan_icfft(N, sg_num, dim; method=:auto) → ICFFTPlan

Construct a standalone backward plan (internally builds forward geometry).
"""
function plan_icfft(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                     method::Symbol=:auto) where D
    ops_s, spec_asu, n_spec, L, M, fill_map, use_centered =
        _plan_geometry(N, sg_num, dim; method)

    if use_centered
        fwd = plan_krfft_centered(spec_asu, ops_s)
        bwd = plan_centered_ikrfft(spec_asu, ops_s, fwd)
    else
        bwd = plan_m2_backward(spec_asu, ops_s)
    end
    return ICFFTPlan(bwd, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
end

# ============================================================================
# plan_cfft_pair — bidirectional
# ============================================================================

"""
    plan_cfft_pair(N, sg_num, dim; method=:auto) → AbstractCFFTPairPlan

Construct a bidirectional CFFT plan (supports both `cfft!` and `icfft!`).

Returns `GeneralCFFTPairPlan` or `CenteredCFFTPairPlan` depending on the
space group centering.
"""
function plan_cfft_pair(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                         method::Symbol=:auto) where D
    ops_s, spec_asu, n_spec, L, M, fill_map, use_centered =
        _plan_geometry(N, sg_num, dim; method)

    if use_centered
        fwd = plan_krfft_centered(spec_asu, ops_s)
        bwd = plan_centered_ikrfft(spec_asu, ops_s, fwd)
        return CenteredCFFTPairPlan(fwd, bwd, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
    else
        fwd = plan_krfft(spec_asu, ops_s)
        bwd = plan_m2_backward(spec_asu, ops_s)
        return GeneralCFFTPairPlan(fwd, bwd, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
    end
end

# ============================================================================
# cfft! — forward transform: f₀(M³) → F̂(n_spec)
# ============================================================================

"""
    cfft!(F̂, plan, f0)

Crystallographic FFT forward transform: `f₀(M³) → F̂(n_spec)`.
Works with `CFFTPlan`, `GeneralCFFTPairPlan`, or `CenteredCFFTPairPlan`.
"""
function cfft! end

# ---- General forward kernel ----
function _cfft_general!(F̂::AbstractVector{ComplexF64},
                         fwd::GeneralForwardPlan,
                         f0::AbstractArray{Float64},
                         M::NTuple{3,Int},
                         n_spec::Int)
    M_vol = prod(M)
    @inbounds @simd for i in 1:M_vol
        fwd.input_buffer[i] = complex(f0[i])
    end
    fft_reconstruct!(fwd)
    @inbounds @simd for i in 1:n_spec
        F̂[i] = fwd.output_buffer[i]
    end
    return F̂
end

# ---- Centered forward kernel ----
function _cfft_centered!(F̂::AbstractVector{ComplexF64},
                          fwd::CenteredForwardPlan,
                          f0::AbstractArray{Float64},
                          n_spec::Int)
    copyto!(fwd.f0_buffer, f0)
    F_out = fft_reconstruct_centered!(fwd)
    @inbounds @simd for i in 1:n_spec
        F̂[i] = F_out[i]
    end
    return F̂
end

# ---- Single-direction CFFTPlan dispatch ----
function cfft!(F̂::AbstractVector{ComplexF64},
               plan::CFFTPlan{<:GeneralForwardPlan},
               f0::AbstractArray{Float64})
    _cfft_general!(F̂, plan.fwd, f0, plan.M, plan.n_spec)
end

function cfft!(F̂::AbstractVector{ComplexF64},
               plan::CFFTPlan{<:CenteredForwardPlan},
               f0::AbstractArray{Float64})
    _cfft_centered!(F̂, plan.fwd, f0, plan.n_spec)
end

# ---- PairPlan dispatch ----
function cfft!(F̂::AbstractVector{ComplexF64},
               plan::GeneralCFFTPairPlan,
               f0::AbstractArray{Float64})
    _cfft_general!(F̂, plan.fwd, f0, plan.M, plan.n_spec)
end

function cfft!(F̂::AbstractVector{ComplexF64},
               plan::CenteredCFFTPairPlan,
               f0::AbstractArray{Float64})
    _cfft_centered!(F̂, plan.fwd, f0, plan.n_spec)
end

# ============================================================================
# icfft! — backward transform: F̂(n_spec) → f₀(M³)
# ============================================================================

"""
    icfft!(f0, plan, F̂)

Crystallographic FFT inverse transform: `F̂(n_spec) → f₀(M³)`.
Works with `ICFFTPlan`, `GeneralCFFTPairPlan`, or `CenteredCFFTPairPlan`.
"""
function icfft! end

# ---- General backward kernel ----
function _icfft_general!(f0::AbstractArray{Float64},
                          bwd::GeneralBackwardPlan,
                          F̂::AbstractVector{ComplexF64},
                          M::NTuple{3,Int})
    M_vol = prod(M)
    f0_buf = execute_m2_backward!(bwd, copy(F̂))
    @inbounds @simd for i in 1:M_vol
        f0[i] = real(f0_buf[i])
    end
    return f0
end

# ---- Centered backward kernel ----
function _icfft_centered!(f0::AbstractArray{Float64},
                           bwd::CenteredBackwardPlan,
                           F̂::AbstractVector{ComplexF64})
    execute_centered_ikrfft!(bwd, copy(F̂), f0)
    return f0
end

# ---- Single-direction ICFFTPlan dispatch ----
function icfft!(f0::AbstractArray{Float64},
                plan::ICFFTPlan{<:GeneralBackwardPlan},
                F̂::AbstractVector{ComplexF64})
    _icfft_general!(f0, plan.bwd, F̂, plan.M)
end

function icfft!(f0::AbstractArray{Float64},
                plan::ICFFTPlan{<:CenteredBackwardPlan},
                F̂::AbstractVector{ComplexF64})
    _icfft_centered!(f0, plan.bwd, F̂)
end

# ---- PairPlan dispatch ----
function icfft!(f0::AbstractArray{Float64},
                plan::GeneralCFFTPairPlan,
                F̂::AbstractVector{ComplexF64})
    _icfft_general!(f0, plan.bwd, F̂, plan.M)
end

function icfft!(f0::AbstractArray{Float64},
                plan::CenteredCFFTPairPlan,
                F̂::AbstractVector{ComplexF64})
    _icfft_centered!(f0, plan.bwd, F̂)
end

# ============================================================================
# Diffusion kernel construction
# ============================================================================

"""
    make_diffusion_kernel(plan::AbstractCFFTPlan, Δs, lattice) → Vector{Float64}

Construct the diffusion kernel `K[i] = exp(-Δs · |k(hᵢ)|²)` for each spectral
ASU point. The caller owns the returned vector.
"""
function make_diffusion_kernel(plan::AbstractCFFTPlan,
                                Δs::Float64,
                                lattice::AbstractMatrix)
    k2 = cfft_k2(plan, lattice)
    return @. exp(-Δs * k2)
end

"""
    update_diffusion_kernel!(K, plan::AbstractCFFTPlan, Δs, lattice)

In-place update of diffusion kernel (when Δs or lattice changes). O(n_spec).
"""
function update_diffusion_kernel!(K::AbstractVector{Float64},
                                   plan::AbstractCFFTPlan,
                                   Δs::Float64,
                                   lattice::AbstractMatrix)
    k2 = cfft_k2(plan, lattice)
    @. K = exp(-Δs * k2)
    return K
end

# ============================================================================
# Spectral geometry
# ============================================================================

"""
    cfft_k2(plan::AbstractCFFTPlan, lattice) → Vector{Float64}

Return `|k(h)|²` for each spectral ASU point, length `n_spec`.
"""
function cfft_k2(plan::AbstractCFFTPlan, lattice::AbstractMatrix)
    N = plan.N
    D = plan.dim
    spec_asu = plan.spec_asu
    n_spec = plan.n_spec

    recip_B = 2π * inv(Matrix(lattice))'

    k2 = Vector{Float64}(undef, n_spec)
    @inbounds for (i, pt) in enumerate(spec_asu.points)
        h_centered = ntuple(d -> pt.idx[d] >= N[d] ÷ 2 ? pt.idx[d] - N[d] : pt.idx[d], Val(D))
        k_vec = recip_B * collect(h_centered)
        k2[i] = dot(k_vec, k_vec)
    end

    return k2
end

# ============================================================================
# Grid conversions
# ============================================================================

"""
    subgrid_to_fullgrid!(f_full, plan::AbstractCFFTPlan, f0)

Expand subgrid M³ data to full grid N³ using symmetry operations (fill_map).
"""
function subgrid_to_fullgrid!(f_full::AbstractArray, plan::AbstractCFFTPlan,
                               f0::AbstractArray)
    fill_map = plan.fill_map
    f0_vec = vec(f0)
    @inbounds for i in eachindex(fill_map)
        f_full[i] = f0_vec[fill_map[i]]
    end
    return f_full
end

"""
    fullgrid_to_subgrid!(f0, plan::AbstractCFFTPlan, f_full)

Extract stride-L subgrid M³ from full grid N³ data.
"""
function fullgrid_to_subgrid!(f0::AbstractArray, plan::AbstractCFFTPlan,
                               f_full::AbstractArray)
    L = plan.L
    M = plan.M
    @inbounds for k in 1:M[3], j in 1:M[2], i in 1:M[1]
        f0[i, j, k] = f_full[(i-1)*L[1]+1, (j-1)*L[2]+1, (k-1)*L[3]+1]
    end
    return f0
end

# ============================================================================
# Query functions
# ============================================================================

"""Return subgrid dimensions M = N ÷ L."""
subgrid_size(plan::AbstractCFFTPlan) = plan.M

"""Return full grid dimensions N."""
fullgrid_size(plan::AbstractCFFTPlan) = plan.N

"""Return stride factors L."""
stride_factors(plan::AbstractCFFTPlan) = plan.L

"""Return number of spectral ASU points."""
cfft_asu_size(plan::AbstractCFFTPlan) = plan.n_spec

end  # module CFFTApi
