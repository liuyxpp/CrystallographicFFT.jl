# ============================================================================
# Public CFFT API — Device-Agnostic
# ============================================================================

module CFFTApi

using LinearAlgebra
using ..SymmetryOps: SymOp, get_ops, detect_centering_type, CentP
using ..ASU: find_optimal_shift
import ..SpectralIndexing as _SpectralIndexingModule
using ..SpectralIndexing: calc_spectral_asu, get_k_vector
const _SpecASU = _SpectralIndexingModule.SpectralIndexing  # the type struct

# Import plan types and functions from parent module scope
# (types.jl, planning.jl, execute.jl are included before this file)
import ..ForwardPlan, ..BackwardPlan
import ..CenteredForwardPlan, ..CenteredBackwardPlan, ..CenteringFoldPlan
import ..RealForwardPlan, ..RealBackwardPlan
import ..auto_L, ..plan_forward, ..plan_backward
import ..plan_centered_forward, ..plan_centered_backward
import ..plan_real_forward, ..plan_real_backward
import ..fft_reconstruct!, ..execute_backward!
import ..fft_reconstruct_centered!, ..execute_centered_backward!
import ..rfft_reconstruct!, ..execute_real_backward!
import ..copy_real_to_complex_kernel!, ..copy_complex_to_real_kernel!

import KernelAbstractions
using KernelAbstractions: get_backend, CPU as KA_CPU

# ── Exports ──────────────────────────────────────────────────────────────────

export AbstractCFFTPlan, AbstractCFFTPairPlan
export GeneralCFFTPairPlan, CenteredCFFTPairPlan
export CFFTPlan, ICFFTPlan
export plan_cfft, plan_icfft, plan_cfft_pair
export cfft!, icfft!
export RCFFTPlan, IRCFFTPlan, GeneralRCFFTPairPlan
export plan_rcfft, plan_ircfft, plan_rcfft_pair
export rcfft!, ircfft!
export make_diffusion_kernel, update_diffusion_kernel!
export cfft_k2
export subgrid_size, fullgrid_size, stride_factors, cfft_asu_size
export subgrid_to_fullgrid!, fullgrid_to_subgrid!

# ── Backend inference (extensible by CUDA extension) ─────────────────────────

"""
    _infer_backend(array_type) → KernelAbstractions backend

Return the KA backend for the given array type.
Defaults to `CPU()`. Overridden by CUDAExt for `CuArray`.
"""
_infer_backend(::Type{<:AbstractArray}) = KA_CPU()
_infer_backend(::Type{<:Array}) = KA_CPU()

# ── Abstract types ───────────────────────────────────────────────────────────

abstract type AbstractCFFTPlan end
abstract type AbstractCFFTPairPlan <: AbstractCFFTPlan end

# ── Plan types ───────────────────────────────────────────────────────────────

"""
    CFFTPlan{FP} <: AbstractCFFTPlan

Forward-only Crystallographic FFT plan. Use with `cfft!`.
"""
struct CFFTPlan{FP, SO<:SymOp} <: AbstractCFFTPlan
    fwd::FP
    ops_shifted::Vector{SO}
    spec_asu::_SpecASU
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
"""
struct ICFFTPlan{BP} <: AbstractCFFTPlan
    bwd::BP
    spec_asu::_SpecASU
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

"""
    GeneralCFFTPairPlan <: AbstractCFFTPairPlan

Bidirectional CFFT plan (General/M2 path).
"""
struct GeneralCFFTPairPlan{FP<:ForwardPlan, BP<:BackwardPlan} <: AbstractCFFTPairPlan
    fwd::FP
    bwd::BP
    spec_asu::_SpecASU
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

Bidirectional CFFT plan (Centered path for I/F/C lattices).
"""
struct CenteredCFFTPairPlan{FP<:CenteredForwardPlan, BP<:CenteredBackwardPlan} <: AbstractCFFTPairPlan
    fwd::FP
    bwd::BP
    spec_asu::_SpecASU
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

"""
    RCFFTPlan{FP} <: AbstractCFFTPlan

Forward-only real CFFT plan using rfft. Use with `rcfft!`.
"""
struct RCFFTPlan{FP, SO<:SymOp} <: AbstractCFFTPlan
    fwd::FP
    ops_shifted::Vector{SO}
    spec_asu::_SpecASU
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

"""
    IRCFFTPlan{BP} <: AbstractCFFTPlan

Backward-only real CFFT plan using irfft. Use with `ircfft!`.
"""
struct IRCFFTPlan{BP} <: AbstractCFFTPlan
    bwd::BP
    spec_asu::_SpecASU
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

"""
    GeneralRCFFTPairPlan <: AbstractCFFTPairPlan

Bidirectional real CFFT plan (General path, rfft/irfft).
"""
struct GeneralRCFFTPairPlan{FP<:RealForwardPlan, BP<:RealBackwardPlan} <: AbstractCFFTPairPlan
    fwd::FP
    bwd::BP
    spec_asu::_SpecASU
    n_spec::Int
    sg_num::Int
    dim::Int
    N::NTuple{3,Int}
    M::NTuple{3,Int}
    L::NTuple{3,Int}
    fill_map::Array{Int32}
end

# ============================================================================
# Shared plan geometry
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

    # Build fill map (symmetry expansion table)
    fill_map = _build_fill_map_internal(shifted_ops, L_vec, collect(M), collect(N), D)

    return shifted_ops, spec_asu, n_spec, L, M, fill_map, use_centered
end

"""Build symmetry fill map: for each full-grid point, its subgrid linear index."""
function _build_fill_map_internal(shifted_ops, L, M_sub, N, D)
    N_vol = prod(N)
    fill_map = zeros(Int32, N...)
    M_sub_tup = Tuple(M_sub)
    li_sub = LinearIndices(M_sub_tup)

    for ci in CartesianIndices(Tuple(N))
        x = [ci[d] - 1 for d in 1:D]  # 0-based

        # Find which subgrid point corresponds to x via symmetry
        found = false
        for op in shifted_ops
            # Apply op: x' = R*x + t mod N
            x_rot = [mod(sum(Int(op.R[d, d2]) * x[d2] for d2 in 1:D) + Int(op.t[d]), N[d])
                      for d in 1:D]

            # Check if x' is on the stride-L subgrid
            if all(mod(x_rot[d], L[d]) == 0 for d in 1:D)
                # Map to subgrid index
                sub_idx = [x_rot[d] ÷ L[d] + 1 for d in 1:D]
                fill_map[ci] = Int32(li_sub[CartesianIndex(Tuple(sub_idx))])
                found = true
                break
            end
        end

        if !found
            # Fallback: map to first subgrid point
            fill_map[ci] = Int32(1)
        end
    end
    return fill_map
end

# ============================================================================
# plan_cfft — forward-only
# ============================================================================

"""
    plan_cfft(N, sg_num, dim; method=:auto, array_type=Array) → CFFTPlan

Construct a forward-only Crystallographic FFT plan.

# Arguments
- `N::NTuple`: Full grid dimensions, e.g. `(64,64,64)`
- `sg_num::Int`: Space group number (1–230)
- `dim::Int`: Spatial dimension (2 or 3)
- `method::Symbol`: `:auto` (default), `:general`, `:centered`
- `array_type`: Array type for backend inference (default: `Array`)
"""
function plan_cfft(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                    method::Symbol=:auto,
                    array_type::Type{<:AbstractArray}=Array) where D
    T = _infer_eltype(array_type)
    backend = _infer_backend(array_type)
    ops_s, spec_asu, n_spec, L, M, fill_map, use_centered =
        _plan_geometry(N, sg_num, dim; method)

    if use_centered
        fwd = plan_centered_forward(T, spec_asu, ops_s; backend)
    else
        fwd = plan_forward(T, spec_asu, ops_s; backend)
    end
    return CFFTPlan(fwd, ops_s, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
end

# ============================================================================
# plan_icfft — backward-only
# ============================================================================

"""
    plan_icfft(fwd_plan::CFFTPlan) → ICFFTPlan

Construct a backward plan from an existing forward plan.
"""
function plan_icfft(fwd_plan::CFFTPlan)
    T = _plan_eltype(fwd_plan)
    ops_s = fwd_plan.ops_shifted
    spec_asu = fwd_plan.spec_asu
    # Infer backend from existing forward plan buffers
    backend = if fwd_plan.fwd isa CenteredForwardPlan
        get_backend(fwd_plan.fwd.f0_buffer)
    else
        get_backend(fwd_plan.fwd.input_buffer)
    end
    if fwd_plan.fwd isa CenteredForwardPlan
        bwd = plan_centered_backward(T, spec_asu, ops_s; backend)
    else
        bwd = plan_backward(T, spec_asu, ops_s; backend)
    end
    return ICFFTPlan(bwd, spec_asu, fwd_plan.n_spec,
                     fwd_plan.sg_num, fwd_plan.dim,
                     fwd_plan.N, fwd_plan.M, fwd_plan.L, fwd_plan.fill_map)
end

"""
    plan_icfft(N, sg_num, dim; method=:auto, array_type=Array) → ICFFTPlan

Construct a standalone backward plan.
"""
function plan_icfft(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                     method::Symbol=:auto,
                     array_type::Type{<:AbstractArray}=Array) where D
    T = _infer_eltype(array_type)
    backend = _infer_backend(array_type)
    ops_s, spec_asu, n_spec, L, M, fill_map, _ =
        _plan_geometry(N, sg_num, dim; method)
    bwd = plan_backward(T, spec_asu, ops_s; backend)
    return ICFFTPlan(bwd, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
end

# ============================================================================
# plan_cfft_pair — bidirectional
# ============================================================================

"""
    plan_cfft_pair(N, sg_num, dim; method=:auto, array_type=Array) → AbstractCFFTPairPlan

Construct a bidirectional CFFT plan.
"""
function plan_cfft_pair(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                         method::Symbol=:auto,
                         array_type::Type{<:AbstractArray}=Array) where D
    T = _infer_eltype(array_type)
    backend = _infer_backend(array_type)
    ops_s, spec_asu, n_spec, L, M, fill_map, use_centered =
        _plan_geometry(N, sg_num, dim; method)

    if use_centered
        fwd = plan_centered_forward(T, spec_asu, ops_s; backend)
        bwd = plan_centered_backward(T, spec_asu, ops_s; backend)
        return CenteredCFFTPairPlan(fwd, bwd, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
    else
        fwd = plan_forward(T, spec_asu, ops_s; backend)
        bwd = plan_backward(T, spec_asu, ops_s; backend)
        return GeneralCFFTPairPlan(fwd, bwd, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
    end
end

# ============================================================================
# cfft! — forward transform
# ============================================================================

"""
    cfft!(F̂, plan, f0)

Crystallographic FFT forward transform: `f₀(M³) → F̂(n_spec)`.
"""
function cfft! end

function cfft!(F̂::AbstractVector{<:Complex},
               plan::CFFTPlan{<:ForwardPlan},
               f0::AbstractArray{<:Real})
    _cfft_general!(F̂, plan.fwd, f0, plan.M, plan.n_spec)
end

function cfft!(F̂::AbstractVector{<:Complex},
               plan::CFFTPlan{<:CenteredForwardPlan},
               f0::AbstractArray{<:Real})
    _cfft_centered!(F̂, plan.fwd, f0, plan.M, plan.n_spec)
end

function cfft!(F̂::AbstractVector{<:Complex},
               plan::GeneralCFFTPairPlan,
               f0::AbstractArray{<:Real})
    _cfft_general!(F̂, plan.fwd, f0, plan.M, plan.n_spec)
end

function cfft!(F̂::AbstractVector{<:Complex},
               plan::CenteredCFFTPairPlan,
               f0::AbstractArray{<:Real})
    _cfft_centered!(F̂, plan.fwd, f0, plan.M, plan.n_spec)
end

function _cfft_general!(F̂, fwd::ForwardPlan, f0, M, n_spec)
    M_vol = prod(M)
    backend = get_backend(fwd.input_buffer)
    if backend isa KA_CPU
        @inbounds @simd for i in 1:M_vol
            fwd.input_buffer[i] = complex(f0[i])
        end
    else
        copy_real_to_complex_kernel!(backend)(
            fwd.input_buffer, f0; ndrange=M_vol)
    end
    fft_reconstruct!(fwd)
    if backend isa KA_CPU
        @inbounds @simd for i in 1:n_spec
            F̂[i] = fwd.output_buffer[i]
        end
    else
        copyto!(F̂, fwd.output_buffer)
    end
    return F̂
end

function _cfft_centered!(F̂, fwd::CenteredForwardPlan, f0, M, n_spec)
    M_vol = prod(M)
    backend = get_backend(fwd.f0_buffer)
    if backend isa KA_CPU
        @inbounds for k in 1:M[3], j in 1:M[2], i in 1:M[1]
            fwd.f0_buffer[i,j,k] = f0[i,j,k]
        end
    else
        copyto!(fwd.f0_buffer, f0)
    end
    fft_reconstruct_centered!(fwd)
    if backend isa KA_CPU
        @inbounds @simd for i in 1:n_spec
            F̂[i] = fwd.krfft_plan.output_buffer[i]
        end
    else
        copyto!(F̂, fwd.krfft_plan.output_buffer)
    end
    return F̂
end

# ============================================================================
# icfft! — backward transform
# ============================================================================

"""
    icfft!(f0, plan, F̂)

Crystallographic FFT inverse transform: `F̂(n_spec) → f₀(M³)`.
"""
function icfft! end

function icfft!(f0::AbstractArray{<:AbstractFloat},
                plan::ICFFTPlan{<:BackwardPlan},
                F̂::AbstractVector{<:Complex})
    _icfft_general!(f0, plan.bwd, F̂, plan.M)
end

function icfft!(f0::AbstractArray{<:AbstractFloat},
                plan::ICFFTPlan{<:CenteredBackwardPlan},
                F̂::AbstractVector{<:Complex})
    _icfft_centered!(f0, plan.bwd, F̂, plan.M)
end

function icfft!(f0::AbstractArray{<:AbstractFloat},
                plan::GeneralCFFTPairPlan,
                F̂::AbstractVector{<:Complex})
    _icfft_general!(f0, plan.bwd, F̂, plan.M)
end

function icfft!(f0::AbstractArray{<:AbstractFloat},
                plan::CenteredCFFTPairPlan,
                F̂::AbstractVector{<:Complex})
    _icfft_centered!(f0, plan.bwd, F̂, plan.M)
end

function _icfft_general!(f0, bwd::BackwardPlan, F̂, M)
    M_vol = prod(M)
    f0_buf = execute_backward!(bwd, F̂)
    backend = get_backend(f0_buf)
    if backend isa KA_CPU
        @inbounds @simd for i in 1:M_vol
            f0[i] = real(f0_buf[i])
        end
    else
        copy_complex_to_real_kernel!(backend)(
            f0, f0_buf; ndrange=M_vol)
    end
    return f0
end

function _icfft_centered!(f0, bwd::CenteredBackwardPlan, F̂, M)
    M_vol = prod(M)
    f0_buf = execute_centered_backward!(bwd, F̂)
    backend = get_backend(f0_buf)
    if backend isa KA_CPU
        @inbounds @simd for i in 1:M_vol
            f0[i] = real(f0_buf[i])
        end
    else
        copy_complex_to_real_kernel!(backend)(
            f0, f0_buf; ndrange=M_vol)
    end
    return f0
end

# ============================================================================
# Diffusion kernel
# ============================================================================

"""Construct diffusion kernel `K[i] = exp(-Δs · |k(hᵢ)|²)`."""
function make_diffusion_kernel(plan::AbstractCFFTPlan,
                                Δs::Float64,
                                lattice::AbstractMatrix)
    k2 = cfft_k2(plan, lattice)
    return @. exp(-Δs * k2)
end

"""In-place update of diffusion kernel."""
function update_diffusion_kernel!(K::AbstractVector,
                                   plan::AbstractCFFTPlan,
                                   Δs::Float64,
                                   lattice::AbstractMatrix)
    k2 = cfft_k2(plan, lattice)
    @. K = exp(-Δs * k2)
    return K
end

"""Return `|k(h)|²` for each spectral ASU point."""
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

"""Expand subgrid M³ to full grid N³ using symmetry fill_map."""
function subgrid_to_fullgrid!(f_full::AbstractArray, plan::AbstractCFFTPlan,
                               f0::AbstractArray)
    fill_map = plan.fill_map
    f0_vec = vec(f0)
    @inbounds for i in eachindex(fill_map)
        f_full[i] = f0_vec[fill_map[i]]
    end
    return f_full
end

"""Extract stride-L subgrid M³ from full grid N³."""
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

subgrid_size(plan::AbstractCFFTPlan) = plan.M
fullgrid_size(plan::AbstractCFFTPlan) = plan.N
stride_factors(plan::AbstractCFFTPlan) = plan.L
cfft_asu_size(plan::AbstractCFFTPlan) = plan.n_spec

# ============================================================================
# Helpers
# ============================================================================

"""Infer float element type from array_type."""
function _infer_eltype(::Type{<:AbstractArray})
    return Float64  # default
end

"""Get T from a plan."""
_plan_eltype(plan::CFFTPlan{<:ForwardPlan{T}}) where T = T
_plan_eltype(plan::CFFTPlan{<:CenteredForwardPlan{T}}) where T = T
_plan_eltype(::CFFTPlan) = Float64
_plan_eltype(plan::RCFFTPlan{<:RealForwardPlan{T}}) where T = T
_plan_eltype(::RCFFTPlan) = Float64

# ============================================================================
# plan_rcfft — forward-only real CFFT
# ============================================================================

"""
    plan_rcfft(N, sg_num, dim; array_type=Array) → RCFFTPlan

Construct a forward-only real CFFT plan using rfft.
Only supports general (P) lattices; for centered lattices use `plan_cfft`.
"""
function plan_rcfft(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                     array_type::Type{<:AbstractArray}=Array) where D
    T = _infer_eltype(array_type)
    backend = _infer_backend(array_type)
    ops_s, spec_asu, n_spec, L, M, fill_map, _ =
        _plan_geometry(N, sg_num, dim; method=:general)

    fwd = plan_real_forward(T, spec_asu, ops_s; backend)
    return RCFFTPlan(fwd, ops_s, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
end

# ============================================================================
# plan_ircfft — backward-only real CFFT
# ============================================================================

"""
    plan_ircfft(fwd_plan::RCFFTPlan) → IRCFFTPlan

Construct a backward plan from an existing real forward plan.
"""
function plan_ircfft(fwd_plan::RCFFTPlan)
    T = _plan_eltype(fwd_plan)
    ops_s = fwd_plan.ops_shifted
    spec_asu = fwd_plan.spec_asu
    backend = get_backend(fwd_plan.fwd.input_buffer)

    bwd = plan_real_backward(T, spec_asu, ops_s; backend)
    return IRCFFTPlan(bwd, spec_asu, fwd_plan.n_spec,
                      fwd_plan.sg_num, fwd_plan.dim,
                      fwd_plan.N, fwd_plan.M, fwd_plan.L, fwd_plan.fill_map)
end

"""
    plan_ircfft(N, sg_num, dim; array_type=Array) → IRCFFTPlan

Construct a standalone backward real CFFT plan.
"""
function plan_ircfft(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                      array_type::Type{<:AbstractArray}=Array) where D
    T = _infer_eltype(array_type)
    backend = _infer_backend(array_type)
    ops_s, spec_asu, n_spec, L, M, fill_map, _ =
        _plan_geometry(N, sg_num, dim; method=:general)
    bwd = plan_real_backward(T, spec_asu, ops_s; backend)
    return IRCFFTPlan(bwd, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
end

# ============================================================================
# plan_rcfft_pair — bidirectional real CFFT
# ============================================================================

"""
    plan_rcfft_pair(N, sg_num, dim; array_type=Array) → GeneralRCFFTPairPlan

Construct a bidirectional real CFFT plan using rfft/irfft.
Only supports general (P) lattices.
"""
function plan_rcfft_pair(N::NTuple{D,Int}, sg_num::Int, dim::Int;
                          array_type::Type{<:AbstractArray}=Array) where D
    T = _infer_eltype(array_type)
    backend = _infer_backend(array_type)
    ops_s, spec_asu, n_spec, L, M, fill_map, _ =
        _plan_geometry(N, sg_num, dim; method=:general)

    fwd = plan_real_forward(T, spec_asu, ops_s; backend)
    bwd = plan_real_backward(T, spec_asu, ops_s; backend)
    return GeneralRCFFTPairPlan(fwd, bwd, spec_asu, n_spec, sg_num, dim, N, M, L, fill_map)
end

# ============================================================================
# rcfft! — forward real transform
# ============================================================================

"""
    rcfft!(F̂, plan, f0)

Real Crystallographic FFT forward transform: `f₀(real M³) → F̂(n_spec)`.
Uses rfft internally for ~2x speedup over cfft!.
"""
function rcfft! end

function rcfft!(F̂::AbstractVector{<:Complex},
                plan::RCFFTPlan{<:RealForwardPlan},
                f0::AbstractArray{<:Real})
    _rcfft_general!(F̂, plan.fwd, f0, plan.n_spec)
end

function rcfft!(F̂::AbstractVector{<:Complex},
                plan::GeneralRCFFTPairPlan,
                f0::AbstractArray{<:Real})
    _rcfft_general!(F̂, plan.fwd, f0, plan.n_spec)
end

function _rcfft_general!(F̂, fwd::RealForwardPlan, f0, n_spec)
    M_vol = prod(fwd.M)
    backend = get_backend(fwd.input_buffer)
    # Copy real input directly (no real→complex conversion needed)
    if backend isa KA_CPU
        @inbounds @simd for i in 1:M_vol
            fwd.input_buffer[i] = f0[i]
        end
    else
        copyto!(fwd.input_buffer, f0)
    end
    rfft_reconstruct!(fwd)
    if backend isa KA_CPU
        @inbounds @simd for i in 1:n_spec
            F̂[i] = fwd.output_buffer[i]
        end
    else
        copyto!(F̂, fwd.output_buffer)
    end
    return F̂
end

# ============================================================================
# ircfft! — backward real transform
# ============================================================================

"""
    ircfft!(f0, plan, F̂)

Real Crystallographic FFT inverse transform: `F̂(n_spec) → f₀(real M³)`.
Uses irfft internally for ~2x speedup over icfft!.
"""
function ircfft! end

function ircfft!(f0::AbstractArray{<:AbstractFloat},
                 plan::IRCFFTPlan{<:RealBackwardPlan},
                 F̂::AbstractVector{<:Complex})
    _ircfft_general!(f0, plan.bwd, F̂)
end

function ircfft!(f0::AbstractArray{<:AbstractFloat},
                 plan::GeneralRCFFTPairPlan,
                 F̂::AbstractVector{<:Complex})
    _ircfft_general!(f0, plan.bwd, F̂)
end

function _ircfft_general!(f0, bwd::RealBackwardPlan, F̂)
    M_vol = prod(bwd.M)
    f0_buf = execute_real_backward!(bwd, F̂)
    backend = get_backend(f0_buf)
    # irfft output is already real — direct copy
    if backend isa KA_CPU
        @inbounds @simd for i in 1:M_vol
            f0[i] = f0_buf[i]
        end
    else
        copyto!(f0, f0_buf)
    end
    return f0
end

end  # module CFFTApi
