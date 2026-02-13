module SymmetryOps

using LinearAlgebra
using Crystalline

export SymOp, apply_op, apply_op!, get_ops, convert_op, check_shift_invariance, dual_ops
export CenteringType, CentP, CentC, CentA, CentI, CentF
export detect_centering_type, strip_centering

struct SymOp
    R::Matrix{Int}
    t::Vector{Int}
end

function convert_op(op::SymOperation, N::Tuple)
    t_grid = op.translation .* collect(N)
    t_int = round.(Int, t_grid)
    if !all(isapprox.(t_grid, t_int, atol=1e-5))
        error("Symmetry operation not commensurate with grid $N")
    end
    return SymOp(round.(Int, op.rotation), t_int)
end

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

function check_shift_invariance(ops::Vector{SymOp}, shift::Vector{Float64}, N::Tuple)
    # Check if shifting origin by `shift` keeps the operations commensurate with grid `N`.
    # Original: g(x) = Rx + t.
    # Shifted: g'(x) = g(x+s)-s = R(x+s)+t-s = Rx + (Rs + t - s).
    # New t' = t + (Rs - s).
    # We return deltas = (Rs - s), so t' = t + deltas.
    # We check if t' is integer. Since t is integer, we need (Rs - s) to be integer?
    # Actually, we need (Rs + t - s) to be integer.
    # Generally (Rs - s) might not be integer, but combined with t it might be?
    # But t is always integer (on grid).
    # So (Rs - s) must be integer.
    deltas = Vector{Vector{Int}}(undef, length(ops))
    deltas = Vector{Vector{Int}}(undef, length(ops))
    for (i, op) in enumerate(ops)
        # We want the change in translation: R*s - s
        delta_float = (op.R * shift .- shift) .* collect(N)
        delta_int = round.(Int, delta_float)
        if !all(isapprox.(delta_float, delta_int, atol=1e-5))
            return false, Vector{Vector{Int}}()
        end
        deltas[i] = delta_int
    end
    return true, deltas
end

"""
    dual_ops(ops::Vector{SymOp}) -> Vector{SymOp}

Compute the dual operations for the reciprocal lattice.
R_dual = (R^{-1})^T
t_dual = 0
"""
function dual_ops(ops::Vector{SymOp})
    dual = Vector{SymOp}()
    for op in ops
        # R is integer matrix with det +/- 1.
        # Inverse of R is exact.
        # We can use float inverse and round, as entries should be integers.
        R_inv = round.(Int, inv(op.R))
        R_dual = transpose(R_inv)
        # Translation in reciprocal space point group is 0
        t_dual = zeros(Int, length(op.t))
        push!(dual, SymOp(collect(R_dual), t_dual))
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
    detect_centering_type(ops::Vector{SymOp}, N::Tuple) → CenteringType

Detect lattice centering from symmetry operations by identifying
pure translations (R = I) with half-cell translation vectors.

Recognizes: P (primitive), C (ab-face), A (bc-face), I (body), F (all-face).
"""
function detect_centering_type(ops::Vector{SymOp}, N::Tuple)
    dim = length(N)
    I_mat = Matrix{Int}(I, dim, dim)

    # Collect fractional translation vectors of pure translations
    pure_translations = NTuple{3,Bool}[]

    for op in ops
        # Check if R is identity
        op.R != I_mat && continue
        all(abs.(op.t) .< 0.1) && continue  # skip zero translation

        # Classify each component as ~N/2 or ~0
        halves = ntuple(dim) do d
            t_frac = abs(op.t[d]) / N[d]
            abs(t_frac - 0.5) < 0.01
        end
        push!(pure_translations, halves)
    end

    isempty(pure_translations) && return CentP

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
    strip_centering(ops::Vector{SymOp}, centering::CenteringType, N::Tuple)
        → (ops_sub::Vector{SymOp}, N_sub::Tuple)

Remove centering translations from ops and remap to the half-grid.

Returns unique point-group operations on the sub-grid `N_sub = N .÷ 2`
(in centered dimensions), with translations reduced `mod N_sub`.

The returned ops are suitable for `plan_krfft` on the sub-grid.
"""
function strip_centering(ops::Vector{SymOp}, centering::CenteringType, N::Tuple)
    dim = length(N)

    # Determine which dimensions are halved
    halve = if centering == CentI || centering == CentF
        ntuple(_ -> true, dim)   # all dimensions halved
    elseif centering == CentC
        ntuple(d -> d <= 2, dim) # x,y halved, z kept
    elseif centering == CentA
        ntuple(d -> d >= 2, dim) # y,z halved, x kept
    else
        ntuple(_ -> false, dim)  # CentP: nothing halved
    end

    N_sub = ntuple(dim) do d
        halve[d] ? N[d] ÷ 2 : N[d]
    end

    # Map each op to sub-grid coordinates: t_sub = mod(t, N_sub)
    seen = Set{Tuple{Matrix{Int}, Vector{Int}}}()
    ops_sub = SymOp[]

    for op in ops
        t_sub = [mod(op.t[d], N_sub[d]) for d in 1:dim]
        key = (op.R, t_sub)
        if !(key in seen)
            push!(seen, key)
            push!(ops_sub, SymOp(op.R, t_sub))
        end
    end

    return ops_sub, N_sub
end

end
