using CrystallographicFFT.ASU
using Crystalline

"""
    compute_full_orbit(p::Vector{Int}, ops::Vector{SymOp}, N::Tuple)

Compute the full orbit of a point p under symmetry operations ops on grid N.
Returns a Set of points.
"""
function compute_full_orbit(p::Vector{Int}, ops::Vector{SymOp}, N::Tuple)
    orbit = Set{Vector{Int}}()
    stack = [p]
    push!(orbit, p)
    
    while !isempty(stack)
        curr = pop!(stack)
        for op in ops
            next_p = apply_op(op, curr, N)
            if !(next_p in orbit)
                push!(orbit, next_p)
                push!(stack, next_p)
            end
        end
    end
    return orbit
end

"""
    validate_asu(asu::Vector{ASUPoint}, ops::Vector{SymOp}, N::Tuple; verbose=true)

Validate that the ASU generates the full grid exactly once (Completeness & Disjointness).
"""
function validate_asu(asu::Vector{ASUPoint}, ops::Vector{SymOp}, N::Tuple; verbose=true)
    total_points = prod(N)
    coverage = Set{Vector{Int}}()
    
    valid = true
    
    for (i, p_asu) in enumerate(asu)
        # Verify the stored index is actually in the ASU list
        p = p_asu.idx
        
        # Compute orbit using the FULL symmetry group
        # Note: The ASU calculation might use a reduced set of ops if optimization allowed it, 
        # but here we must use the full group defining the symmetry.
        orb = compute_full_orbit(p, ops, N)
        
        # Check Disjointness
        intersection = intersect(coverage, orb)
        if !isempty(intersection)
            if verbose
                println("❌ Overlap detected at ASU point $p (index $i)")
                println("   Overlapping points: $(collect(intersection)[1:min(5, end)])")
            end
            valid = false
        end
        
        union!(coverage, orb)
    end
    
    # Check Completeness
    covered_count = length(coverage)
    if covered_count != total_points
        if verbose
            println("❌ Incomplete coverage!")
            println("   Expected: $total_points")
            println("   Actual:   $covered_count")
            println("   Missing:  $(total_points - covered_count) points")
        end
        valid = false
    else
        if verbose
            println("✅ Completeness Verified: $covered_count / $total_points")
        end
    end
    
    if valid && verbose
        println("✅ ASU Validation Passed!")
    end
    
    return valid
end

function run_validation(sg_num, dim, N, name)
    println("="^40)
    println("Validating $name (SG $sg_num, $dim-D, Grid $N)")
    ops = get_ops(sg_num, dim, N)
    asu = calc_asu(N, ops)
    println("ASU Size: $(length(asu))")
    
    is_valid = validate_asu(asu, ops, N)
    if !is_valid
        error("Validation failed for $name")
    end
end

# --- RUN CHECKS ---

# 1. p2mm (SG 6, 2D)
run_validation(6, 2, (8, 8), "p2mm")

# 2. p2mg (SG 7, 2D)
run_validation(7, 2, (8, 8), "p2mg")

# 3. Pmmm (SG 47, 3D)
# 8x8x8 grid checks 3D recursion logic
# 4. p2mm (SG 6, 2D) - 16x16 (Check depth scaling)
run_validation(6, 2, (16, 16), "p2mm_16")
