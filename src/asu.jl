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
    
    # The optimal shift fraction (e.g. 0.5, 1/3) depends only on space group
    # symmetry, not on N. Find it at small proxy N, then apply to target N.
    # Proxy N must be divisible by shift denominators (2,3,4,5).
    # Use 60 = LCM(2,3,4,5) to cover all candidates.
    N_proxy = ntuple(_ -> 60, D)  # 60³=216K points — fast for orbit enum
    ops_proxy = _rescale_ops(ops, N, N_proxy)
    
    frac_candidates = [
        zeros(Float64, D),
        fill(0.5, D),
        fill(1.0/3.0, D),
        fill(0.25, D),
        fill(0.2, D),
    ]
    
    best_frac = frac_candidates[1]
    min_sp_count = typemax(Int)
    
    for frac in frac_candidates
        shift = frac ./ collect(N_proxy)
        valid, deltas = check_shift_invariance(ops_proxy, shift, N_proxy)
        !valid && continue
        
        curr_ops_proxy = [SymOp(op.R, op.t .+ deltas[i]) for (i, op) in enumerate(ops_proxy)]
        sp_count = _count_special_positions_fast(N_proxy, curr_ops_proxy)
        
        if sp_count < min_sp_count
            min_sp_count = sp_count
            best_frac = frac
        end
        min_sp_count == 0 && break
    end
    
    # Apply best_frac to the actual N
    shift = best_frac ./ collect(N)
    valid, deltas = check_shift_invariance(ops, shift, N)
    if valid
        best_ops = [SymOp(op.R, op.t .+ deltas[i]) for (i, op) in enumerate(ops)]
        return shift, best_ops
    else
        return zeros(Float64, D), ops
    end
end

"""
    _rescale_ops(ops, N_from, N_to) -> Vector{SymOp}

Rescale translation vectors from grid N_from to grid N_to.
R stays the same, t_new = t * (N_to / N_from).
"""
function _rescale_ops(ops::Vector{SymOp}, N_from::Tuple, N_to::Tuple)
    D = length(N_from)
    result = Vector{SymOp}(undef, length(ops))
    for (i, op) in enumerate(ops)
        t_new = [round(Int, op.t[d] * N_to[d] ÷ N_from[d]) for d in 1:D]
        result[i] = SymOp(op.R, t_new)
    end
    return result
end

"""
    _count_special_positions_fast(N, ops) -> Int

Count the number of ASU points with orbit multiplicity < |G|.
Uses direct orbit enumeration with BitArray visited mask.
O(N³ × |G|) with no recursion.
"""
function _count_special_positions_fast(N::Tuple, ops::Vector{SymOp})
    D = length(N)
    n_total = prod(N)
    n_ops = length(ops)
    max_orbit = n_ops  # Maximum orbit size = |G|
    
    visited = falses(n_total)
    sp_count = 0
    n_asu = 0
    
    x = zeros(Int, D)
    x_rot = zeros(Int, D)
    R_mats = [op.R for op in ops]
    t_vecs = [op.t for op in ops]
    
    for lin_idx in 1:n_total
        visited[lin_idx] && continue
        
        # Convert linear index to coordinate
        rem = lin_idx - 1
        @inbounds for d in 1:D
            x[d] = rem % N[d]
            rem = rem ÷ N[d]
        end
        
        # Compute orbit via worklist
        worklist = [lin_idx]
        visited[lin_idx] = true
        wi = 1
        
        while wi <= length(worklist)
            curr_lin = worklist[wi]
            wi += 1
            
            rem_c = curr_lin - 1
            @inbounds for d in 1:D
                x_rot[d] = rem_c % N[d]
                rem_c = rem_c ÷ N[d]
            end
            
            for g in 1:n_ops
                R = R_mats[g]
                t = t_vecs[g]
                @inbounds begin
                    li = 0
                    stride = 1
                    for d in 1:D
                        s = t[d]
                        for j in 1:D
                            s += R[d, j] * x_rot[j]
                        end
                        li += mod(s, N[d]) * stride
                        stride *= N[d]
                    end
                end
                li += 1
                
                if !visited[li]
                    visited[li] = true
                    push!(worklist, li)
                end
            end
        end
        
        orbit_size = length(worklist)
        n_asu += 1
        if orbit_size < max_orbit
            sp_count += 1
        end
    end
    
    return sp_count
end

function calc_asu(N::Tuple, ops::Vector{SymOp})
    D = length(N)
    asu_points = Vector{ASUPoint}()

    # Queue: (N, ops, scale, offset, depth, is_gp)
    queue = Any[(N, ops, ones(Int, D), zeros(Int, D), zeros(Int, D), falses(D))]

    # Pre-allocate reusable buffers for apply_op!
    tmp_out = zeros(Int, D)
    p_even = zeros(Int, D)
    p_odd = zeros(Int, D)

    while !isempty(queue)
        (curr_N, curr_ops, curr_scale, curr_offset, curr_depth, curr_gp) = pop!(queue)

        if any(x->x==0, curr_N); continue; end

        if all(curr_gp) || all(x->x<=1, curr_N)
            # Leaf: Generate points & orbits
            local_pts = vec(collect(Iterators.product([0:n-1 for n in curr_N]...)))
            visited = Set{NTuple{D, Int}}()

            for p_tuple in local_pts
                p_tuple in visited && continue
                p = collect(p_tuple)

                orbit_tuples = Set{NTuple{D, Int}}([p_tuple])
                stack = [p]
                while !isempty(stack)
                    curr_p = pop!(stack)
                    for op in curr_ops
                        apply_op!(tmp_out, op, curr_p, curr_N)
                        next_tuple = NTuple{D, Int}(tmp_out)
                        if !(next_tuple in orbit_tuples)
                            push!(orbit_tuples, next_tuple)
                            push!(stack, copy(tmp_out))
                        end
                    end
                end

                rep = collect(minimum(orbit_tuples))
                push!(asu_points, ASUPoint(curr_scale .* rep .+ curr_offset, curr_depth, length(orbit_tuples)))
                union!(visited, orbit_tuples)
            end
            continue
        end

        # Split step
        active_dims = findall(.!curr_gp)
        effective_gp = copy(curr_gp)

        # Check even/odd preservation — use apply_op! to avoid allocation
        for d in active_dims
            p_even .= 0; p_odd .= 0; p_odd[d] = 1
            parity_ok = true
            for op in curr_ops
                apply_op!(tmp_out, op, p_even, curr_N)
                if tmp_out[d] % 2 != 0
                    parity_ok = false; break
                end
                apply_op!(tmp_out, op, p_odd, curr_N)
                if tmp_out[d] % 2 != 1
                    parity_ok = false; break
                end
            end
            if !parity_ok
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
                # In-place multiply for R*parity
                for i in 1:D
                    tmp_out[i] = op.t[i] - parity[i]
                    for j in 1:D
                        tmp_out[i] += op.R[i, j] * parity[j]
                    end
                end

                if any(R_val .% S_diag .!= 0) || any(tmp_out .% S_diag .!= 0)
                    valid_sector = false; break
                end
                push!(new_ops, SymOp(R_val .÷ S_diag, tmp_out .÷ S_diag))
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
