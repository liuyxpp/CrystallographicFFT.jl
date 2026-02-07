module ASU

export calc_asu, ASUPoint, classify_points, ASUBlock, CrystallographicASU, pack_asu, find_optimal_shift, InterleavedOrbit, analyze_interleaved_orbits, pack_asu_interleaved
# Re-export SymOp related from SymmetryOps
using ..SymmetryOps
export SymOp, apply_op, get_ops

using LinearAlgebra
using Crystalline

"""
    InterleavedOrbit

Represents an orbit of subgrids (ASUBlocks) under the symmetry group.
"""
struct InterleavedOrbit
    representative::Vector{Int}       # The x_s (shift) stored in memory
    members::Vector{Vector{Int}}      # All x_s in this orbit
    ops::Vector{Vector{SymOp}}        # Operations mapping rep -> member (one list per member)
    multiplicity::Int                 # Size of the orbit (unique members)
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
    orbit::Union{Nothing, InterleavedOrbit} # Added for Mode B
end

# Backward compatibility constructor
ASUBlock(data::A, range::Vector{StepRange{Int, Int}}, depth::Vector{Int}) where {T, N, A<:AbstractArray{T, N}} = ASUBlock(data, range, depth, nothing)


struct CrystallographicASU{D, T, A}
    dim_blocks::Dict{Int, Vector{ASUBlock{T, D, A}}}
    shift::NTuple{D, Float64}
end



"""
    calc_asu(sg_num, dim, N::Tuple) -> (points, shift)
High-level entry point with automatic Magic Shift optimization.
"""
function calc_asu(sg_num, dim, N::Tuple)
    ops = get_ops(sg_num, dim, N)
    best_shift, shifted_ops = find_optimal_shift(ops, N)
    return calc_asu(N, shifted_ops), Tuple(best_shift)
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




"""
    analyze_interleaved_orbits(N::Tuple, ops::Vector{SymOp}; L::Union{Nothing, Tuple}=nothing)

Analyze the symmetry orbits of the subgrids defined by stride `L`.
If `L` is not provided, defaults to `(2, 2, ...)` (Standard Cooley-Tukey Radix-2).
Returns a list of `InterleavedOrbit`s, which become the `ASUBlock`s.
"""
function analyze_interleaved_orbits(N::Tuple, ops::Vector{SymOp}; L::Union{Nothing, Tuple}=nothing)
    D = length(N)
    if isnothing(L)
        L = Tuple(fill(2, D))
    end
    
    # Generate all coset shifts (coarse grid points)
    # 0 : L-1
    coarse_pts = vec(collect(Iterators.product([0:l-1 for l in L]...)))
    visited = Set{Vector{Int}}()
    orbits = Vector{InterleavedOrbit}()
    
    # Pre-calculate integer operations for speed?
    # R is usually integer. t is fractional.
    # We need t_grid = round(t * N).
    
    for p_tuple in coarse_pts
        p = collect(p_tuple)
        if p in visited
            continue
        end
        
        # Start new orbit
        member_set = Set([p])
        members = [p]
        ops_map = Dict{Vector{Int}, Vector{SymOp}}()
        ops_map[p] = [SymOp(Matrix{Int}(I, D, D), zeros(Int, D))] # Identity maps p to p
        
        # BFS to find all connected subgrids
        queue = [p]
        while !isempty(queue)
            curr = popfirst!(queue)
            
            for op in ops
                # Calculate transformed shift index
                # x' = (R * x + t_grid) % L
                # t_grid = op.t (already in grid units!)
                # Be careful with floating point t.
                
                t_grid = op.t
                x_prime = (op.R * curr .+ t_grid) .% L
                
                # Handle negative modulo in Julia
                x_prime = mod.(x_prime, L)
                
                x_prime_vec = Vector(x_prime)
                
                if !(x_prime_vec in member_set)
                    push!(member_set, x_prime_vec)
                    push!(members, x_prime_vec)
                    push!(queue, x_prime_vec)
                    ops_map[x_prime_vec] = [op]
                else
                    # Add op to existing member if needed?
                    # For simplicty, we just need ONE op to map rep -> member.
                    # But keeping all might be useful for verification.
                    if haskey(ops_map, x_prime_vec)
                        push!(ops_map[x_prime_vec], op)
                    else
                        ops_map[x_prime_vec] = [op]
                    end
                end
            end
        end
        
        # Store orbit
        # We need ops mapping Representative (p) -> Member (m)
        # The BFS above mapped curr -> next. 
        # Actually, for a group, applying all ops to 'p' generates the orbit directly.
        # We don't need BFS if 'ops' is just generators, we need BFS.
        # 'get_ops' usually returns the full group (including expanded translations for centered cells?).
        # Let's assume 'ops' is the full set of operations relevant for the ASU.
        
        # Re-scan to build exact mapping p -> m
        final_members = sort(collect(member_set))
        final_ops = Vector{Vector{SymOp}}(undef, length(final_members))
        
        for (i, m) in enumerate(final_members)
            # Find all ops g such that g(p) = m
            matching_ops = SymOp[]
            for op in ops
                t_grid = op.t
                x_prime = mod.((op.R * p .+ t_grid), L)
                if x_prime == m
                    push!(matching_ops, op)
                end
            end
            final_ops[i] = matching_ops
        end
        
        push!(orbits, InterleavedOrbit(p, final_members, final_ops, length(final_members)))
        union!(visited, member_set)
    end
    
    return orbits
end

include("pack_asu.jl")

end
