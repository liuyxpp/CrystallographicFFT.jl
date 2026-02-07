module SymmetryOps

using LinearAlgebra
using Crystalline

export SymOp, apply_op, get_ops, convert_op, check_shift_invariance, dual_ops

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

function apply_op(op::SymOp, x::Vector{Int}, N::Tuple)
    map((val, mod) -> mod1(val + 1, mod) - 1, op.R * x .+ op.t, N)
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

end
