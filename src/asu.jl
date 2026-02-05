module ASU

export SymOp, apply_op, calc_asu, ASUPoint, classify_points, get_ops
export ASUBlock, CrystallographicASU, pack_asu, find_optimal_shift

using LinearAlgebra
using Crystalline

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

struct ASUPoint
    idx::Vector{Int}
    depth::Vector{Int}
    multiplicity::Int
end

struct ASUBlock{T, N, A<:AbstractArray{T, N}}
    data::A
    range::Vector{StepRange{Int, Int}}
    depth::Vector{Int}
end

struct CrystallographicASU{D, T, A}
    dim_blocks::Dict{Int, Vector{ASUBlock{T, D, A}}}
    shift::NTuple{D, Float64}
end

include("pack_asu.jl")

"""
    calc_asu(sg_num, dim, N::Tuple) -> (points, shift)
High-level entry point with automatic Magic Shift optimization.
"""
function calc_asu(sg_num, dim, N::Tuple)
    ops = get_ops(sg_num, dim, N)
    best_shift, shifted_ops = find_optimal_shift(ops, N)
    return calc_asu(N, shifted_ops), Tuple(best_shift)
end

function check_shift_invariance(ops::Vector{SymOp}, shift::Vector{Float64}, N::Tuple)
    deltas = Vector{Vector{Int}}(undef, length(ops))
    for (i, op) in enumerate(ops)
        delta_float = (op.R * shift .+ (op.t ./ collect(N)) .- shift) .* collect(N)
        delta_int = round.(Int, delta_float)
        if !all(isapprox.(delta_float, delta_int, atol=1e-5))
            return false, Vector{Vector{Int}}()
        end
        deltas[i] = delta_int
    end
    return true, deltas
end

function find_optimal_shift(ops::Vector{SymOp}, N::Tuple)
    D = length(N)
    candidates = [
        zeros(Float64, D),
        fill(0.5, D) ./ collect(N),
        fill(1.0/3.0, D) ./ collect(N),
        fill(0.25, D) ./ collect(N),
        fill(0.2, D) ./ collect(N)
    ]

    best_shift, best_ops = zeros(Float64, D), ops
    min_sp_count = typemax(Int)

    for cand in candidates
        valid, deltas = check_shift_invariance(ops, cand, N)
        !valid && continue

        curr_ops = [SymOp(op.R, op.t .+ deltas[i]) for (i, op) in enumerate(ops)]
        points = calc_asu(N, curr_ops)

        max_mult = maximum(p -> p.multiplicity, points)
        sp_count = count(p -> p.multiplicity < max_mult, points)

        if sp_count < min_sp_count
            min_sp_count = sp_count
            best_shift = cand
            best_ops = curr_ops
        end
        min_sp_count == 0 && break
    end
    return best_shift, best_ops
end

function calc_asu(N::Tuple, ops::Vector{SymOp})
    D = length(N)
    asu_points = Vector{ASUPoint}()

    # Queue: (N, ops, scale, offset, depth, is_gp)
    queue = Any[(N, ops, ones(Int, D), zeros(Int, D), zeros(Int, D), falses(D))]

    while !isempty(queue)
        (curr_N, curr_ops, curr_scale, curr_offset, curr_depth, curr_gp) = pop!(queue)

        if any(x->x==0, curr_N); continue; end

        if all(curr_gp) || all(x->x<=1, curr_N)
            # Leaf: Generate points & orbits
            local_pts = vec(collect(Iterators.product([0:n-1 for n in curr_N]...)))
            visited = Set{Vector{Int}}()

            for p_tuple in local_pts
                p = collect(p_tuple)
                p in visited && continue

                orbit = Set([p])
                stack = [p]
                while !isempty(stack)
                    curr_p = pop!(stack)
                    for op in curr_ops
                        next_p = apply_op(op, curr_p, curr_N)
                        if !(next_p in orbit)
                            push!(orbit, next_p); push!(stack, next_p)
                        end
                    end
                end

                push!(asu_points, ASUPoint(curr_scale .* sort(collect(orbit))[1] .+ curr_offset, curr_depth, length(orbit)))
                union!(visited, orbit)
            end
            continue
        end

        # Split step
        active_dims = findall(.!curr_gp)
        effective_gp = copy(curr_gp)

        # Check even/odd preservation
        for d in active_dims
            p_even, p_odd = zeros(Int, D), zeros(Int, D); p_odd[d] = 1
            if any(op -> apply_op(op, p_even, curr_N)[d]%2 != 0 || apply_op(op, p_odd, curr_N)[d]%2 != 1, curr_ops)
                effective_gp[d] = true
            end
        end

        # Process sectors
        ranges = [effective_gp[d] || curr_N[d] <= 1 ? (0:0) : (0:1) for d in 1:D]

        for parity in Iterators.product(ranges...)
            parity = collect(parity)
            S_diag = [effective_gp[d] ? 1 : 2 for d in 1:D]

            # Filter valid ops for this sector
            new_ops = Vector{SymOp}()
            valid_sector = true

            for op in curr_ops
                # R_new = S\R*S, t_new = S\(R*p + t - p)
                R_val = op.R .* transpose(S_diag)
                t_val = op.R * parity .+ op.t .- parity

                if any(R_val .% S_diag .!= 0) || any(t_val .% S_diag .!= 0)
                    valid_sector = false; break
                end
                push!(new_ops, SymOp(R_val .÷ S_diag, t_val .÷ S_diag))
            end

            !valid_sector && continue

            # Next state
            next_N = [effective_gp[d] ? curr_N[d] : curr_N[d] ÷ 2 for d in 1:D]
            next_gp = [effective_gp[d] || (parity[d]==1 && !effective_gp[d]) for d in 1:D] # Odd becomes GP leaf
            next_depth = [effective_gp[d] ? curr_depth[d] : curr_depth[d] + (parity[d]==0) for d in 1:D]

            push!(queue, (tuple(next_N...), new_ops, curr_scale .* S_diag, curr_scale .* parity .+ curr_offset, next_depth, next_gp))
        end
    end
    sort!(asu_points, by = p->p.idx)
end

end
