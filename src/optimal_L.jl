"""
Optimal L computation for KRFFT Mode B.

This module provides functions to compute the optimal decomposition factor L
for each space group, using Crystalline.jl to obtain symmetry information.
Supports both isotropic and anisotropic L values.
"""

using Crystalline

export optimal_L, optimal_L_isotropic, recommended_N, optimal_L_phase1

using ..SymmetryOps: SymOp, get_ops
using ..ASU: find_optimal_shift

"""
    optimal_L_phase1(ops_shifted::Vector{SymOp}) -> (L, n_active, n_reachable)

Determine optimal Phase 1 Cooley-Tukey factor L from shifted symmetry operations.

For each dimension d, L_d = 2 if any shifted operation has odd translation t_d,
otherwise L_d = 1 (no subgrid parity flip available in that dimension).

# Returns
- `L::Vector{Int}`: Optimal L per dimension (each 1 or 2)
- `n_active::Int`: Number of independent FFTs needed = prod(L) / n_reachable
- `n_reachable::Int`: Number of distinct subgrids reachable from subgrid 0

# Algorithm
The 2x2y2z Cooley-Tukey decomposition with A₀ = diag(L) splits each dimension by
L_d, creating prod(L) subgrids indexed by parity (n₁, n₂, ..., nD) ∈ {0,1}^D.

A symmetry operation Sg with rotation R_g and translation t_g maps subgrid 0
(even-indexed points: x = L·x₁) to a subgrid determined by:

    parity = t_g mod L

So the set of subgrids reachable from subgrid 0 is {t_g mod L | g ∈ G}.

If all prod(L) subgrids are reachable (n_reachable = prod(L)), only one
independent FFT is needed (n_active = 1) and the theoretical speedup is prod(L).
Otherwise, n_active = prod(L) / n_reachable independent FFTs are needed.

# Examples
```julia
# Pmmm (SG 47): 3 mirrors → all 8 subgrids reachable
ops = get_ops(47, 3, (16,16,16))
_, ops_s = find_optimal_shift(ops, (16,16,16))
L, n_active, n_reach = optimal_L_phase1(ops_s)
# L = [2,2,2], n_active = 1, n_reach = 8

# Pmm2 (SG 25): 2 mirrors → only 4 subgrids reachable
ops = get_ops(25, 3, (16,16,16))
_, ops_s = find_optimal_shift(ops, (16,16,16))
L, n_active, n_reach = optimal_L_phase1(ops_s)
# L = [2,2,1], n_active = 1, n_reach = 4
```
"""
function optimal_L_phase1(ops_shifted::Vector{SymOp})
    dim = length(ops_shifted[1].t)
    
    # Step 1: Determine L per dimension from translation parity
    L = ones(Int, dim)
    for op in ops_shifted
        t = round.(Int, op.t)
        for d in 1:dim
            if mod(t[d], 2) == 1
                L[d] = 2
            end
        end
    end
    
    # Step 2: Count reachable subgrids from subgrid 0
    n_subgrids = prod(L)
    if n_subgrids == 1
        return L, 1, 1  # No factorization possible
    end
    
    reachable = Set{Vector{Int}}()
    for op in ops_shifted
        t = round.(Int, op.t)
        parity = [mod(t[d], L[d]) for d in 1:dim]
        push!(reachable, parity)
    end
    n_reachable = length(reachable)
    
    # Number of independent FFTs = number of cosets
    n_active = n_subgrids ÷ n_reachable
    
    # Step 3: If n_active > 1, consider reducing L to eliminate wasted subgrids
    # E.g., P-1 (|G|=2) gives L=(2,2,2) but only 2 reachable → n_active=4
    # Better: find the minimal L such that n_reachable = prod(L), 
    # giving n_active = 1 and speedup = prod(L)
    if n_active > 1
        L, n_active, n_reachable = _optimize_L_for_coverage(ops_shifted, L, dim)
    end
    
    return L, n_active, n_reachable
end

"""
    optimal_L_phase1(sg::Int, dim::Int, N::Tuple) -> (L, n_active, n_reachable)

Convenience method that gets operations and applies find_optimal_shift internally.
"""
function optimal_L_phase1(sg::Int, dim::Int, N::Tuple)
    ops = get_ops(sg, dim, N)
    _, ops_shifted = find_optimal_shift(ops, N)
    return optimal_L_phase1(ops_shifted)
end

"""
Find the largest subset of L dimensions where all subgrids are reachable.
Tries all 2^D subsets of dimensions and picks the one with maximum product.
"""
function _optimize_L_for_coverage(ops::Vector{SymOp}, L_max::Vector{Int}, dim::Int)
    best_product = 1
    best_L = ones(Int, dim)
    best_active = 1
    best_reach = 1
    
    # Try all subsets of dimensions to set L_d = 2
    for mask in 0:(2^dim - 1)
        L_try = ones(Int, dim)
        for d in 1:dim
            if (mask >> (d-1)) & 1 == 1 && L_max[d] == 2
                L_try[d] = 2
            end
        end
        
        n_sub = prod(L_try)
        if n_sub == 1
            continue
        end
        
        # Count reachable subgrids with this L
        reachable = Set{Vector{Int}}()
        for op in ops
            t = round.(Int, op.t)
            parity = [mod(t[d], L_try[d]) for d in 1:dim]
            push!(reachable, parity)
        end
        n_reach = length(reachable)
        n_active = n_sub ÷ n_reach
        
        # Prefer: n_active=1 with maximum product
        # If tied, prefer fewer active blocks
        speedup = n_sub ÷ n_active  # = n_reach, actual speedup per FFT
        current_best_speedup = prod(best_L) ÷ best_active
        
        if n_active == 1 && n_sub > best_product ÷ max(best_active, 1)
            best_L = L_try
            best_product = n_sub
            best_active = 1
            best_reach = n_reach
        elseif n_active == 1 && best_active > 1
            # Any n_active=1 solution beats n_active>1
            best_L = L_try
            best_product = n_sub
            best_active = 1
            best_reach = n_reach
        end
    end
    
    return best_L, best_active, best_reach
end

"""
    optimal_L(sg::Int, D::Int=3) -> Tuple{Int, ...}

Compute the optimal decomposition factor L for space group `sg` in `D` dimensions.

For anisotropic crystal systems (orthorhombic, tetragonal, monoclinic), 
L can be different in each direction to accommodate different cell edge lengths.

# Returns
- `L::NTuple{D, Int}`: Optimal L in each direction (Lx, Ly, Lz)

# Algorithm
1. Get point group operations from Crystalline.jl
2. Analyze rotation matrices to find cycle lengths for each axis pair
3. Compute L[d] = lcm of cycles affecting axis d
4. Ensure L ≥ 3 to avoid stabilization by 2-fold operations

# Examples
```julia
julia> optimal_L(47, 3)  # Pmmm (orthorhombic)
(4, 4, 4)

julia> optimal_L(75, 3)  # P4 (tetragonal)
(4, 4, 2)  # z-axis has no rotation coupling
```
"""
function optimal_L(sg::Int, D::Int=3)
    # Get point group operations from Crystalline.jl
    pg_ops = _get_point_group_rotations(sg, D)
    
    # Analyze rotation cycles for each axis
    axis_cycles = _analyze_axis_cycles(pg_ops, D)
    
    # Compute L per axis
    L = Tuple(_compute_L_for_axis(axis_cycles[d]) for d in 1:D)
    
    return L
end

"""
    optimal_L_isotropic(sg::Int, D::Int=3) -> Int

Compute the isotropic (same in all directions) optimal L for space group `sg`.

This is the LCM of all rotation cycle lengths in the point group,
with a minimum of 3 to avoid 2-fold stabilization.
"""
function optimal_L_isotropic(sg::Int, D::Int=3)
    pg_ops = _get_point_group_rotations(sg, D)
    all_cycles = _get_all_rotation_cycles(pg_ops)
    
    if isempty(all_cycles)
        return 2  # No rotations, any L works
    end
    
    L = lcm(all_cycles...)
    
    # L=2 is sufficient — KRFFT's b-shift handles mirrors/inversions
    return max(L, 2)
end

"""
    recommended_N(sg::Int, target_N::NTuple{D, Int}) -> NTuple{D, Int}

Given target grid dimensions, return adjusted N that enables optimal KRFFT speedup.

Adjusts each dimension independently to be a multiple of the corresponding optimal L.
"""
function recommended_N(sg::Int, target_N::NTuple{D, Int}) where D
    L = optimal_L(sg, D)
    
    # Round each dimension up to nearest multiple of L[d]
    adjusted_N = Tuple(
        L[d] * ceil(Int, target_N[d] / L[d]) 
        for d in 1:D
    )
    
    return adjusted_N
end

"""
    group_order(sg::Int, D::Int=3) -> Int

Return the order |G| of the space group `sg` (number of symmetry operations).

This includes all operations with distinct translations (e.g., body centering, 
glide planes), not just the point group rotations.

For Ia3d (sg=230): |G| = 96 (48 point group × 2 body centering translations).
"""
function group_order(sg::Int, D::Int=3)
    SG = spacegroup(sg, Val(D))
    return length(SG)
end

# ========== Internal Implementation ==========

"""Get point group rotation matrices from Crystalline.jl."""
function _get_point_group_rotations(sg::Int, D::Int)
    # Use Crystalline.jl to get space group
    SG = spacegroup(sg, Val(D))
    
    # Extract rotation matrices (point group part)
    rotations = Matrix{Int}[]
    for op in SG
        R = rotation(op)
        # Convert to standard matrix format
        push!(rotations, Matrix{Int}(R))
    end
    
    # Remove duplicates (same rotation with different translations)
    unique_rotations = unique(rotations)
    
    return unique_rotations
end

"""Analyze rotation cycles affecting each axis."""
function _analyze_axis_cycles(rotations::Vector{Matrix{Int}}, D::Int)
    # For each axis, collect cycle lengths of rotations affecting it
    axis_cycles = [Int[] for _ in 1:D]
    
    for R in rotations
        if R == I(D)
            continue  # Skip identity
        end
        
        # Determine rotation order
        order = _rotation_order(R)
        
        # Determine which axes are affected
        # A rotation affects axis d if R changes coordinates in that dimension
        for d in 1:D
            if _axis_affected_by_rotation(R, d, D)
                push!(axis_cycles[d], order)
            end
        end
        
        # Check for mirrors/inversions (det = -1)
        if det(R) < 0
            # Inversion/mirror affects all flipped axes
            for d in 1:D
                if R[d, d] == -1
                    push!(axis_cycles[d], 2)  # Will be upgraded to 4
                end
            end
        end
    end
    
    return axis_cycles
end

"""Compute rotation order (smallest n such that R^n = I)."""
function _rotation_order(R::Matrix{Int})
    D = size(R, 1)
    Rn = copy(R)
    for n in 1:12  # Max order in crystallography is 6
        if Rn ≈ I(D)
            return n
        end
        Rn = Rn * R
    end
    return 1  # Fallback
end

"""Check if rotation R affects axis d."""
function _axis_affected_by_rotation(R::Matrix{Int}, d::Int, D::Int)
    # Axis d is affected if R mixes coordinate d with others,
    # or if R[d,d] ≠ 1
    
    if R[d, d] != 1
        return true
    end
    
    # Check off-diagonal elements in row and column d
    for i in 1:D
        if i != d
            if R[d, i] != 0 || R[i, d] != 0
                return true
            end
        end
    end
    
    return false
end

"""Get all unique rotation cycle lengths."""
function _get_all_rotation_cycles(rotations::Vector{Matrix{Int}})
    cycles = Int[]
    D = size(rotations[1], 1)
    
    for R in rotations
        if R == I(D)
            continue
        end
        
        order = _rotation_order(R)
        push!(cycles, order)
        
        # Mirror/inversion counts as 2
        if det(R) < 0
            push!(cycles, 2)
        end
    end
    
    return unique(cycles)
end

"""Compute L for a single axis given its cycle lengths."""
function _compute_L_for_axis(cycles::Vector{Int})
    if isempty(cycles)
        return 2  # No constraints, minimum L
    end
    
    # LCM of all cycles
    L = lcm(cycles...)
    
    # KRFFT approach: L=2 is sufficient for mirrors/inversions.
    # The origin shift b=(1/2,...) makes all translations distinct mod 2,
    # so we do NOT need to bump L from 2 to 4.
    # See KRFFT II §4.2: "2x2y2z" algorithm for Pmmm uses A₀=2I with b-shift.
    
    return max(L, 2)  # Minimum L=2
end

# ========== Utility Functions ==========

"""
    print_optimal_L_table(groups::Vector{Int}=common_groups())

Print a reference table of optimal L values.
"""
function print_optimal_L_table(groups::Vector{Int}=[1, 2, 47, 75, 143, 168, 195, 230])
    println("=" ^ 70)
    println("Space Group Optimal L Reference Table (from Crystalline.jl)")
    println("=" ^ 70)
    println()
    println("SG#     |G|    L (isotropic)   L (anisotropic)   N constraint")
    println("-" ^ 70)
    
    for sg in groups
        try
            G_order = group_order(sg)
            L_iso = optimal_L_isotropic(sg)
            L_aniso = optimal_L(sg, 3)
            constraint = "N = $(L_iso)k"
            
            println("$(lpad(sg, 3))   $(lpad(G_order, 4))   $(lpad(L_iso, 15))   $(lpad(string(L_aniso), 15))   $constraint")
        catch e
            println("$(lpad(sg, 3))   ERROR: $e")
        end
    end
    
    println()
end
