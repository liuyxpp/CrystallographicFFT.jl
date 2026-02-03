module ASU

export SymOp, apply_op, calc_asu, ASUPoint, classify_points, get_ops

using LinearAlgebra
using Crystalline

"""
    convert_op(op::SymOperation, N::Tuple)

Convert a Crystalline.jl SymOperation to an ASU.SymOp for a grid of size N.
Checks for commensurability (translation must be integer steps on the grid).
"""
function convert_op(op::SymOperation, N::Tuple)
    # Rotation: convert to Int Matrix
    R = round.(Int, op.rotation)
    
    # Translation: convert fractional to grid units
    # t_frac = op.translation
    # t_grid = t_frac .* N
    
    t_grid = op.translation .* collect(N)
    t_int = round.(Int, t_grid)
    
    # Check commensurability error
    if !all(isapprox.(t_grid, t_int, atol=1e-5))
        error("Symmetry operation not commensurate with grid $N. Translation: $(op.translation), Grid Error: $(t_grid .- t_int)")
    end
    
    # We must normalize fractional translations to [0, 1) before scaling? 
    # Crystalline usually provides canonical translations.
    # But SymOp logic in calc_asu uses modulo arithmetic on result.
    # So t_int doesn't strictly need to be in [0, N), but applied as (Rx + t).
    
    return SymOp(R, t_int)
end

"""
    get_ops(sg_num::Int, dim::Int, N::Tuple)

Get symmetry operations for a space group (by number and dimension) adapted for grid N.
"""
function get_ops(sg_num::Int, dim::Int, N::Tuple)
    sg = spacegroup(sg_num, dim)
    ops = operations(sg)
    return [convert_op(op, N) for op in ops]
end

struct SymOp
    R::Matrix{Int}
    t::Vector{Int}
end

function apply_op(op::SymOp, x::Vector{Int}, N::Tuple)
    # x is 0-indexed
    # x_new = (R * x + t) % N
    x_new = (op.R * x .+ op.t)
    # Handle negative modulo correctly
    return map((val, mod) -> mod1(val + 1, mod) - 1, x_new, N)
end

struct ASUPoint
    idx::Vector{Int}    # Global 0-based indices
    depth::Vector{Int}  # Recursion depth per dimension (0 = Odd/GP, k = k-th Even split)
    multiplicity::Int   # Size of the orbit (approximate/local)
end

struct ASUBlock{T, N, A<:AbstractArray{T, N}}
    data::A
    range::Vector{StepRange{Int, Int}} # Global indices
    depth::Vector{Int}
end

struct CrystallographicASU{D, T, A}
    dim_blocks::Dict{Int, Vector{ASUBlock{T, D, A}}}
end

export SymOp, apply_op, calc_asu, ASUPoint, classify_points, get_ops
export ASUBlock, CrystallographicASU, pack_asu

function calc_asu(sg, dim, N::Tuple)
    ops = get_ops(sg, dim, N)
    return calc_asu(N, ops)
end

"""
    calc_asu(N::Tuple, ops::Vector{SymOp})

Calculate the Crystallographic Asymmetric Unit (ASU) using the recursive parity filter algorithm.
"""
function calc_asu(N::Tuple, ops::Vector{SymOp})
    D = length(N)
    asu_points = Vector{ASUPoint}()
    
    # State: (current_N, current_ops, logical_to_global_map, current_depth, is_gp)
    # logical_to_global_map: x_global = A * x_local + b
    # represented by scale and offset. x_global = scale .* x_local .+ offset
    
    initial_scale = ones(Int, D)
    initial_offset = zeros(Int, D)
    initial_depth = zeros(Int, D)
    initial_gp = falses(D)
    
    queue = []
    push!(queue, (N, ops, initial_scale, initial_offset, initial_depth, initial_gp))
    
    while !isempty(queue)
        (curr_N, curr_ops, curr_scale, curr_offset, curr_depth, curr_gp) = pop!(queue)
        
        if any(n -> n == 0, curr_N)
            continue
        end

        # Check if we should stop recursion
        # Stop if all dimensions are marked as GP (General Position) or N<=1
        if all(curr_gp) || all(n -> n <= 1, curr_N)
            # Leaf node: Compute orbits for the current grid
            # Generate all points in the current local grid
            local_points = vec(collect(Iterators.product([0:n-1 for n in curr_N]...)))
            
            # Use Union-Find or visited set to find orbits
            visited = Set{Vector{Int}}()
            
            for p_tuple in local_points
                p = collect(p_tuple)
                if p in visited
                    continue
                end
                
                # Generate orbit
                orbit = Set{Vector{Int}}()
                stack = [p]
                push!(orbit, p)
                
                while !isempty(stack)
                    curr_p = pop!(stack)
                    for op in curr_ops
                        next_p = apply_op(op, curr_p, curr_N)
                        if !(next_p in orbit)
                            push!(orbit, next_p)
                            push!(stack, next_p)
                        end
                    end
                end
                
                # Pick representative (lexicographically first)
                rep = sort(collect(orbit))[1]
                
                # Mark all in orbit as visited
                union!(visited, orbit)
                
                # Calculate global coordinate of the rep
                global_idx = curr_scale .* rep .+ curr_offset
                
                # Store
                push!(asu_points, ASUPoint(global_idx, curr_depth, length(orbit)))
            end
            
            continue
        end
        
        # Recursion Step: Split non-GP dimensions
        # Determine strict even/odd domains for active dimensions
        active_dims = findall(.!curr_gp)
        
        # Check splittability: Dimensions must not be coupled (mixed parity) by Ops
        # If a dimension mixes Even/Odd, we cannot split it (must treat as GP/Leaf for that dim)
        effective_gp = copy(curr_gp)
        
        for d in active_dims
            is_splittable = true
            # Test point 0 (Even)
            p_even = zeros(Int, D)
            # Test point 1 (Odd) - we need to construct vector with 1 at d
            p_odd = zeros(Int, D); p_odd[d] = 1
            
            for op in curr_ops
                # Check Even -> Even
                # val = (R*0 + t) % N = t % N
                # We need val % 2 == 0
                val_even = op.t[d] # apply_op logic handles modulo, but parity check is local
                # Wait, apply_op includes modulo N.
                # If N=2, t=1. 1%2 != 0.
                # Use apply_op to be safe
                res_even = apply_op(op, p_even, curr_N)[d]
                if res_even % 2 != 0
                    is_splittable = false; break
                end
                
                # Check Odd -> Odd
                res_odd = apply_op(op, p_odd, curr_N)[d]
                if res_odd % 2 != 1
                    is_splittable = false; break
                end
            end
            
            if !is_splittable
                effective_gp[d] = true
            end
        end

        # Iterate over all parity combinations for active dimensions
        # For inactive (GP) dimensions, we just keep the single "0" parity (no split)
        ranges = []
        for d in 1:D
            if effective_gp[d] || curr_N[d] <= 1
                push!(ranges, [0]) # Dummy parity
            else
                push!(ranges, [0, 1]) # 0=Even, 1=Odd
            end
        end
        
        sectors = Iterators.product(ranges...)
        
        for sector_parity in sectors
            sector_parity = collect(sector_parity)
            # sector_parity[d] is 0 (Even) or 1 (Odd)
            
            # Valid Sector Check: Does the group map this sector to itself?
            # We check the mapping of the "zero" point of the sector (offset)
            # adjusted for parity.
            # Actually, simply construct the transformation and check consistency.
            
            # Construct new mapping parameters
            # Even (0): x_old = 2 * x_new + 0
            # Odd  (1): x_old = 2 * x_new + 1
            # If GP (frozen), x_old = x_new (scale doesn't change from prev, but here we view it as x_old = 1*x_new + 0)
            
            # Wait, our logic for GP is that we effectively "consumed" the parity.
            # So if GP, treating as identity is fine.
            
            # We need to verify if Ops satisfy the parity constraint.
            # Sample point: x_local = 0 for all dims.
            # x_sector_local = sector_parity (since 2*0 + p = p)
            # Apply op: x'_sector_local = op(x_sector_local)
            # Check if x'_sector_local has same parity as sector_parity.
            
            is_closed = true
            
            # Check a few points or derive algebraically.
            # Algebra: new_val = R(2x+p) + t.
            # We need new_val % 2 == p for all x?
            # Or rather, new_val % 2 should be constant for the sector?
            # If it maps to a different sector, we have cross-coupling.
            # In CFFT design, we assume they don't cross-couple or handled specifically.
            # For p2mm/p2mg, we know they are closed.
            # If they are NOT closed, we should structurally unite them.
            # For this feasibility, throw error or skip if not closed? 
            # Or just assume closed as per design doc availability.
            # Let's check closure of the offset p.
            
            # Let's derive the new ops first.
            # Relation: p_old = S * p_new + offset
            # S is diagonal matrix. S_ii = 2 if !GP, 1 if GP.
            # offset_i = parity_i.
            
            # New Op (R', t'):
            # p_old' = R p_old + t
            # S p_new' + off = R (S p_new + off) + t
            # S p_new' = R S p_new + R off + t - off
            # p_new' = S^-1 R S p_new + S^-1 (R off + t - off)
            
            # Check integer integrality.
            # This implicitly checks closure.
            
            new_ops = Vector{SymOp}()
            valid_sector = true
            
            S_diag = [effective_gp[d] ? 1 : 2 for d in 1:D]
            offset = sector_parity
            
            for op in curr_ops
                # R_new = S^-1 R S
                # component wise: R_new_ij = R_ij * S_j / S_i
                
                R_new = zeros(Int, D, D)
                t_new = zeros(Int, D)
                
                # Check R integrality
                for i in 1:D, j in 1:D
                    val = op.R[i,j] * S_diag[j]
                    if val % S_diag[i] != 0
                        valid_sector = false
                        break
                    end
                    R_new[i,j] = val ÷ S_diag[i]
                end
                if !valid_sector break end
                
                # Check t integrality
                # t_term = (R * offset + t - offset)
                t_term = op.R * offset .+ op.t .- offset
                
                for i in 1:D
                    if t_term[i] % S_diag[i] != 0
                        valid_sector = false
                        break
                    end
                    t_new[i] = t_term[i] ÷ S_diag[i]
                end
                
                push!(new_ops, SymOp(R_new, t_new))
            end


            
            if !valid_sector
                continue
            end
            
            # Prepare next state
            next_gp = copy(effective_gp)
            next_depth = copy(curr_depth)
            next_N = zeros(Int, D)
            
            for d in 1:D
                if !effective_gp[d]
                    if curr_N[d] <= 1
                        # We are at N=1, so we can't split further. 
                        # Treat as fixed/GP for subsequent recursions.
                        next_gp[d] = true
                        next_N[d] = 1
                    elseif sector_parity[d] == 1 # Odd
                        next_gp[d] = true # Mark as General Position / Stop split
                        next_N[d] = curr_N[d] ÷ 2 # Odd count is N/2
                    else # Even
                        next_gp[d] = false # Join the next recursion
                        next_N[d] = curr_N[d] ÷ 2 # Even count is N/2
                        next_depth[d] += 1
                    end
                else
                    # GP stays GP
                    next_N[d] = curr_N[d] # Count preserved
                end
            end
            
            # Update map
            # x_global = old_scale * x_old + old_offset
            # x_old = S * x_new + p
            # x_global = old_scale * (S * x_new + p) + old_offset
            #          = (old_scale * S) * x_new + (old_scale * p + old_offset)
            
            next_scale = curr_scale .* S_diag
            next_offset = curr_scale .* sector_parity .+ curr_offset
            
            push!(queue, (tuple(next_N...), new_ops, next_scale, next_offset, next_depth, next_gp))
        end
    end
    
    return asu_points
end

function classify_points(points::Vector{ASUPoint})
    # Group by logic
    # Just return sorted list for now
    sort(points, by = p->p.idx)
end

include("pack_asu.jl")

end
