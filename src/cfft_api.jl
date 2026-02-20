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
export cfft_k2, cfft_kk_orbsum
export subgrid_size, fullgrid_size, stride_factors, cfft_asu_size
export subgrid_to_fullgrid!, fullgrid_to_subgrid!
export SubgridStarMap, build_subgrid_star_map, expand_stars!, compress_stars!

# ── Backend inference (extensible by CUDA extension) ─────────────────────────

"""
    _infer_backend(array_type) → KernelAbstractions backend

Return the KA backend for the given array type.
Defaults to `CPU()`. Overridden by CUDAExt for `CuArray`.
"""
_infer_backend(::Type{<:AbstractArray}) = KA_CPU()
_infer_backend(::Type{<:Array}) = KA_CPU()

# ── SubgridStarMap ───────────────────────────────────────────────────────────

"""
    SubgridStarMap

Mapping between subgrid linear indices and star (symmetry orbit) indices.
Stars are the orbits of subgrid points under the space group symmetry
projected onto the stride-L subgrid.

# Fields
- `sub_to_star`: subgrid linear index → star index (length M_vol)
- `star_offsets`: CSR row offsets (length n_stars+1)
- `star_sub_list`: CSR column list — subgrid indices in each star (length M_vol)
- `sub_degeneracy`: number of subgrid points in each star (length n_stars)
- `n_stars`: total number of distinct stars
"""
struct SubgridStarMap
    sub_to_star::Vector{Int32}
    star_offsets::Vector{Int32}
    star_sub_list::Vector{Int32}
    sub_degeneracy::Vector{Int32}
    n_stars::Int
end

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
# Spectral geometry: cfft_kk_orbsum
# ============================================================================

"""
    cfft_kk_orbsum(plan, lattice) → Vector{Vector{Float64}}

Return orbit-summed kα·kβ tensor components (Voigt order) for each spectral
ASU point.  The 6 components are: xx, yy, zz, yz, xz, xy.

For each ASU representative hᵢ, we enumerate its orbit under the reciprocal
point group, compute k(h) = B*·h for every orbit member, accumulate
kα·kβ, and normalise by the orbit size.

This is used for computing the stress tensor directly in the spectral ASU.
"""
function cfft_kk_orbsum(plan::AbstractCFFTPlan, lattice::AbstractMatrix)
    N = plan.N
    D = plan.dim
    spec_asu = plan.spec_asu
    n_spec = plan.n_spec

    recip_B = 2π * inv(Matrix(lattice))'

    # Voigt order: xx, yy, zz, yz, xz, xy
    voigt_pairs = [(1,1), (2,2), (3,3), (2,3), (1,3), (1,2)]
    n_voigt = 6
    kk = [Vector{Float64}(undef, n_spec) for _ in 1:n_voigt]

    # Reciprocal point group operations
    recip_ops = spec_asu.ops
    n_ops = length(recip_ops)
    R_mats = [op.R for op in recip_ops]

    h_buf = zeros(Int, D)
    k_rot = zeros(Int, D)
    k_vec = zeros(Float64, D)
    h_centered = zeros(Float64, D)

    for (spec_idx, pt) in enumerate(spec_asu.points)
        h_asu = pt.idx  # 0-based k-vector
        orbit_size = pt.multiplicity

        # Accumulate kk tensor over orbit
        kk_acc = zeros(Float64, n_voigt)

        # Track visited orbit members to avoid double-counting
        visited = Set{NTuple{D,Int}}()

        for g in 1:n_ops
            R = R_mats[g]
            # h' = R * h_asu mod N
            @inbounds for d in 1:D
                s = 0
                for j in 1:D
                    s += R[d, j] * h_asu[j]
                end
                k_rot[d] = mod(s, N[d])
            end
            h_key = NTuple{D,Int}(Tuple(k_rot))
            h_key in visited && continue
            push!(visited, h_key)

            # Convert to centered frequency and physical k
            @inbounds for d in 1:D
                h_centered[d] = k_rot[d] >= N[d] ÷ 2 + 1 ? k_rot[d] - N[d] : k_rot[d]
            end
            mul!(k_vec, recip_B, h_centered)

            # Accumulate Voigt components
            @inbounds for (v, (α, β)) in enumerate(voigt_pairs)
                kk_acc[v] += k_vec[α] * k_vec[β]
            end
        end

        # Normalise by orbit size
        @inbounds for v in 1:n_voigt
            kk[v][spec_idx] = kk_acc[v] / orbit_size
        end
    end

    return kk
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
# Star conversions (SubgridStarMap)
# ============================================================================

"""
    build_subgrid_star_map(plan::AbstractCFFTPlan) → SubgridStarMap

Build the mapping between subgrid linear indices and stars (symmetry orbits).
A star groups subgrid points that are related by the space group symmetry.

The returned map supports efficient `expand_stars!` (star → subgrid) and
`compress_stars!` (subgrid → star) conversions.
"""
function build_subgrid_star_map(plan::AbstractCFFTPlan)
    M = plan.M
    N = plan.N
    L = plan.L
    D = plan.dim

    M_vol = prod(M)
    li_sub = LinearIndices(M)

    # Union-Find for grouping subgrid points into stars
    parent = collect(Int32(1):Int32(M_vol))

    function uf_find(x::Int32)::Int32
        while parent[x] != x
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        end
        return x
    end

    function uf_union!(a::Int32, b::Int32)
        ra = uf_find(a)
        rb = uf_find(b)
        if ra != rb
            if ra < rb
                parent[rb] = ra
            else
                parent[ra] = rb
            end
        end
    end

    # For each subgrid point, apply all symmetry ops. If the result
    # lands on the subgrid (x' mod L == 0), the two subgrid points
    # belong to the same star. Union-Find groups them.
    ops_s = _get_ops_shifted(plan)

    for ci in CartesianIndices(M)
        sub_idx1 = Int32(li_sub[ci])
        # Subgrid point in full-grid coordinates (0-based)
        x0 = ntuple(d -> (ci[d] - 1) * L[d], Val(D))

        for op in ops_s
            # Apply op: x' = R * x0 + t mod N
            on_subgrid = true
            sub_coords = ntuple(Val(D)) do d
                s = Int(op.t[d])
                for j in 1:D
                    s += Int(op.R[d, j]) * x0[j]
                end
                xp = mod(s, N[d])
                if xp % L[d] != 0
                    on_subgrid = false
                end
                xp ÷ L[d] + 1
            end

            if on_subgrid
                sub_idx2 = Int32(li_sub[CartesianIndex(sub_coords)])
                uf_union!(sub_idx1, sub_idx2)
            end
        end
    end

    # Compress roots and assign star indices
    # Finalize path compression
    for i in Int32(1):Int32(M_vol)
        parent[i] = uf_find(Int32(i))
    end

    # Map roots to sequential star indices
    root_to_star = Dict{Int32, Int32}()
    star_count = Int32(0)
    sub_to_star = Vector{Int32}(undef, M_vol)
    for i in 1:M_vol
        root = parent[i]
        if !haskey(root_to_star, root)
            star_count += 1
            root_to_star[root] = star_count
        end
        sub_to_star[i] = root_to_star[root]
    end
    n_stars = Int(star_count)

    # Compute degeneracy (points per star)
    sub_degeneracy = zeros(Int32, n_stars)
    for i in 1:M_vol
        sub_degeneracy[sub_to_star[i]] += 1
    end

    # Build CSR structure
    star_offsets = Vector{Int32}(undef, n_stars + 1)
    star_offsets[1] = 1
    for s in 1:n_stars
        star_offsets[s + 1] = star_offsets[s] + sub_degeneracy[s]
    end

    star_sub_list = Vector{Int32}(undef, M_vol)
    cursor = copy(star_offsets)  # working cursor per star
    for i in 1:M_vol
        s = sub_to_star[i]
        star_sub_list[cursor[s]] = Int32(i)
        cursor[s] += 1
    end

    return SubgridStarMap(sub_to_star, star_offsets, star_sub_list,
                          sub_degeneracy, n_stars)
end

# Helper: extract shifted ops from any plan type
function _get_ops_shifted(plan::CFFTPlan)
    return plan.ops_shifted
end
function _get_ops_shifted(plan::RCFFTPlan)
    return plan.ops_shifted
end
function _get_ops_shifted(plan::AbstractCFFTPlan)
    # Fallback: recompute from sg_num
    ops = get_ops(plan.sg_num, plan.dim, plan.N)
    _, ops_s = find_optimal_shift(ops, plan.N)
    return ops_s
end

"""
    expand_stars!(f0, map::SubgridStarMap, compressed)

Expand star values to subgrid: `f0[i] = compressed[star(i)]`.
"""
function expand_stars!(f0::AbstractArray, map::SubgridStarMap,
                       compressed::AbstractVector)
    f0_vec = vec(f0)
    @inbounds @simd for i in eachindex(f0_vec)
        f0_vec[i] = compressed[map.sub_to_star[i]]
    end
    return f0
end

"""
    compress_stars!(compressed, map::SubgridStarMap, f0)

Compress subgrid to star values by averaging points within each star.
"""
function compress_stars!(compressed::AbstractVector, map::SubgridStarMap,
                         f0::AbstractArray)
    fill!(compressed, zero(eltype(compressed)))
    f0_vec = vec(f0)
    @inbounds for i in eachindex(f0_vec)
        compressed[map.sub_to_star[i]] += f0_vec[i]
    end
    @inbounds @simd for s in 1:map.n_stars
        compressed[s] /= map.sub_degeneracy[s]
    end
    return compressed
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
Supports all space groups (general and centered lattices).
Always uses the general reconstruction path internally.
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
Supports all space groups (general and centered lattices).
Always uses the general reconstruction path internally.
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
