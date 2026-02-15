module SymmetryOps

using LinearAlgebra
using Crystalline
using StaticArrays

export SymOp, apply_op, apply_op!, get_ops, convert_op, check_shift_invariance, dual_ops
export CenteringType, CentP, CentC, CentA, CentI, CentF
export detect_centering_type, strip_centering

"""
    SymOp{D}

Symmetry operation with rotation matrix R and translation vector t,
parametrized by spatial dimension D. Uses stack-allocated StaticArrays.
"""
struct SymOp{D}
    R::SMatrix{D, D, Int}
    t::SVector{D, Int}
end

# Convenience constructors: auto-convert from regular Matrix/Vector
# Exclude SMatrix/SVector to avoid infinite recursion with the inner constructor
function SymOp(R::Matrix{<:Integer}, t::AbstractVector{<:Integer})
    D = size(R, 1)
    @assert size(R) == (D, D) "R must be square"
    @assert length(t) == D "t must have same dimension as R"
    SymOp{D}(SMatrix{D,D,Int}(R), SVector{D,Int}(t))
end

# Accept Float64 / Real inputs (round to Int) — used by fractal_krfft.jl
function SymOp(R::AbstractMatrix{<:Real}, t::AbstractVector{<:Real})
    SymOp(Matrix{Int}(round.(Int, R)), Vector{Int}(round.(Int, t)))
end



function convert_op(op::SymOperation{D}, N::NTuple{D, Int}) where {D}
    t_grid = op.translation .* collect(N)
    t_int = round.(Int, t_grid)
    if !all(isapprox.(t_grid, t_int, atol=1e-5))
        error("Symmetry operation not commensurate with grid $N")
    end
    return SymOp{D}(SMatrix{D,D,Int}(round.(Int, op.rotation)),
                    SVector{D,Int}(t_int))
end

# Allow Tuple input (non-NTuple dispatch)
convert_op(op::SymOperation{D}, N::Tuple) where {D} = convert_op(op, NTuple{D,Int}(N))

function get_ops(sg_num::Int, dim::Int, N::Tuple)
    [convert_op(op, N) for op in operations(spacegroup(sg_num, dim))]
end

"""
    apply_op!(out, op, x, N)

In-place version: write result of applying `op` to `x` into `out`.
Avoids all allocations.
"""
function apply_op!(out::Vector{Int}, op::SymOp, x::Vector{Int}, N::Tuple)
    D = length(N)
    R = op.R
    t = op.t
    @inbounds for i in 1:D
        s = t[i]
        for j in 1:D
            s += R[i, j] * x[j]
        end
        out[i] = mod(s, N[i])
    end
    return out
end

function apply_op(op::SymOp, x::Vector{Int}, N::Tuple)
    out = similar(x)
    apply_op!(out, op, x, N)
    return out
end

function check_shift_invariance(ops::Vector{<:SymOp}, shift::Vector{Float64}, N::Tuple)
    D = length(N)
    deltas = Vector{SVector{D,Int}}(undef, length(ops))
    for (i, op) in enumerate(ops)
        # We want the change in translation: R*s - s
        delta_float = (Matrix(op.R) * shift .- shift) .* collect(N)
        delta_int = round.(Int, delta_float)
        if !all(isapprox.(delta_float, delta_int, atol=1e-5))
            return false, SVector{D,Int}[]
        end
        deltas[i] = SVector{D,Int}(delta_int)
    end
    return true, deltas
end

"""
    dual_ops(ops::Vector{<:SymOp}) -> Vector{SymOp}

Compute the dual operations for the reciprocal lattice.
R_dual = (R^{-1})^T
t_dual = 0
"""
function dual_ops(ops::Vector{<:SymOp{D}}) where {D}
    dual = Vector{SymOp{D}}(undef, length(ops))
    z = zero(SVector{D,Int})
    for (i, op) in enumerate(ops)
        R_inv = round.(Int, inv(Matrix(op.R)))
        R_dual = SMatrix{D,D,Int}(R_inv')
        dual[i] = SymOp{D}(R_dual, z)
    end
    return dual
end

# ============================================================================
# Centering Type Detection & Stripping
# ============================================================================

"""
    CenteringType

Enumeration of Bravais lattice centering types.
"""
@enum CenteringType CentP CentC CentA CentI CentF

"""
    detect_centering_type(ops::Vector{<:SymOp}, N::Tuple) → CenteringType

Detect lattice centering from symmetry operations by identifying
pure translations (R = I) with half-cell translation vectors.

Recognizes: P (primitive), C (ab-face), A (bc-face), I (body), F (all-face).
"""
function detect_centering_type(ops::Vector{<:SymOp{D}}, N::Tuple) where {D}
    I_mat = SMatrix{D,D,Int}(I)

    # Collect fractional translation vectors of pure translations
    pure_translations = NTuple{D,Bool}[]

    for op in ops
        # Check if R is identity
        op.R != I_mat && continue
        all(d -> abs(op.t[d]) < 0.1, 1:D) && continue  # skip zero translation

        # Classify each component as ~N/2 or ~0
        halves = ntuple(D) do d
            t_frac = abs(op.t[d]) / N[d]
            abs(t_frac - 0.5) < 0.01
        end
        push!(pure_translations, halves)
    end

    isempty(pure_translations) && return CentP

    # Only 3D centering classification
    D < 3 && return CentP

    has_xyz = any(t -> all(t), pure_translations)  # (½,½,½)
    has_0yz = any(t -> !t[1] && t[2] && t[3], pure_translations)  # (0,½,½)
    has_x0z = any(t -> t[1] && !t[2] && t[3], pure_translations)  # (½,0,½)
    has_xy0 = any(t -> t[1] && t[2] && !t[3], pure_translations)  # (½,½,0)

    # F-centering: has at least 2 of {(0,½,½), (½,0,½), (½,½,0)}
    n_face = has_0yz + has_x0z + has_xy0
    n_face >= 2 && return CentF

    # I-centering: (½,½,½)
    has_xyz && return CentI

    # C-centering: (½,½,0) only
    has_xy0 && return CentC

    # A-centering: (0,½,½) only
    has_0yz && return CentA

    # B-centering (½,0,½) — rare, treated similarly
    has_x0z && return CentA

    return CentP
end

"""
    strip_centering(ops::Vector{<:SymOp}, centering::CenteringType, N::Tuple)
        → (ops_sub::Vector{SymOp}, N_sub::Tuple)

Remove centering translations from ops and remap to the half-grid.

Returns unique point-group operations on the sub-grid `N_sub = N .÷ 2`
(in centered dimensions), with translations reduced `mod N_sub`.

The returned ops are suitable for `plan_krfft` on the sub-grid.
"""
function strip_centering(ops::Vector{<:SymOp{D}}, centering::CenteringType, N::Tuple) where {D}
    # Determine which dimensions are halved
    halve = if centering == CentI || centering == CentF
        ntuple(_ -> true, D)   # all dimensions halved
    elseif centering == CentC
        ntuple(d -> d <= 2, D) # x,y halved, z kept
    elseif centering == CentA
        ntuple(d -> d >= 2, D) # y,z halved, x kept
    else
        ntuple(_ -> false, D)  # CentP: nothing halved
    end

    N_sub = ntuple(D) do d
        halve[d] ? N[d] ÷ 2 : N[d]
    end

    # Map each op to sub-grid coordinates: t_sub = mod(t, N_sub)
    seen = Set{Tuple{SMatrix{D,D,Int}, SVector{D,Int}}}()
    ops_sub = SymOp{D}[]

    for op in ops
        t_sub = SVector{D,Int}(ntuple(d -> mod(op.t[d], N_sub[d]), D))
        key = (op.R, t_sub)
        if !(key in seen)
            push!(seen, key)
            push!(ops_sub, SymOp{D}(op.R, t_sub))
        end
    end

    return ops_sub, N_sub
end

end
