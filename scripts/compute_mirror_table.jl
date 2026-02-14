#!/usr/bin/env julia
"""
Precompute the mirror/glide adaptation table for all 230 3D space groups.

For each space group, this script:
1. Extracts all symmetry operations from Crystalline.jl
2. Identifies mirror/glide operations (det(R) = -1, R² = I)
3. Finds the best set of ≤3 mutually orthogonal mirror/glide planes
4. Computes the basis transformation matrix T (if non-standard basis needed)
5. Detects centering type (P/I/F/C/A/B)
6. Outputs a Julia source file with the precomputed table

Usage:
    cd /home/lyx/Develop/CrystallographicFFT.jl
    julia --project scripts/compute_mirror_table.jl
"""

using Crystalline
using LinearAlgebra

# ============================================================================
# Helper: Check if an operation is a mirror/glide (reflection + optional translation)
# A mirror/glide has R such that:
#   - det(R) = -1
#   - R² = I  (the rotation part is an involution)
# The mirror plane normal is the eigenvector with eigenvalue -1.
# ============================================================================

function is_mirror_or_glide(R::Matrix{Int})
    D = size(R, 1)
    # Check R² = I
    R2 = R * R
    for i in 1:D, j in 1:D
        if R2[i,j] != (i == j ? 1 : 0)
            return false
        end
    end
    # Check det(R) = -1
    return round(Int, det(R)) == -1
end

"""
Get the normal vector of a mirror/glide plane from its rotation matrix.
The normal is the eigenvector with eigenvalue -1.
Since R is integer and R² = I with det=-1, exactly one eigenvalue is -1
(in 3D: eigenvalues are -1, +1, +1).
"""
function mirror_normal(R::Matrix{Int})
    D = size(R, 1)
    # R + I has rank D-1, and the null space of (R + I) is the eigenspace for -1.
    # But we want the eigenspace for -1, which is the null space of R - (-I) = R + I.
    RpI = R + I  # R + I: columns in null space have eigenvalue -1
    # Actually: Rv = -v  ⟹  (R+I)v = 0
    # So normal is in null(R+I)
    
    # For 3D integer matrices, we can find this directly:
    # The null space of R+I is 1-dimensional (since det(R)=-1 and R²=I → eigenvalues -1,+1,+1)
    
    # Use cross product of the rows of R+I (they span a 2D space, their cross product is normal)
    if D == 3
        r1 = RpI[1, :]
        r2 = RpI[2, :]
        r3 = RpI[3, :]
        # Cross product of two linearly independent rows
        n = cross(r1, r2)
        if all(n .== 0)
            n = cross(r1, r3)
        end
        if all(n .== 0)
            n = cross(r2, r3)
        end
        if all(n .== 0)
            # Fallback: R+I = 0, meaning R = -I (inversion), not a mirror
            return nothing
        end
        # Normalize to primitive integer vector
        g = gcd(abs.(n)...)
        if g > 0
            n = n .÷ g
        end
        return n
    elseif D == 2
        # 2D: R+I is a 2×2 matrix with rank 1. Normal is perpendicular to its rows.
        r1 = RpI[1, :]
        if all(r1 .== 0)
            r1 = RpI[2, :]
        end
        if all(r1 .== 0)
            return nothing
        end
        n = [-r1[2], r1[1]]
        g = gcd(abs.(n)...)
        if g > 0
            n = n .÷ g
        end
        return n
    end
    return nothing
end

"""
Check if two integer vectors are parallel (same or opposite direction).
"""
function are_parallel(n1::Vector{Int}, n2::Vector{Int})
    # Check if n1 × n2 = 0 (3D) or n1[1]*n2[2] - n1[2]*n2[1] = 0 (2D)
    D = length(n1)
    if D == 3
        c = cross(n1, n2)
        return all(c .== 0)
    elseif D == 2
        return n1[1]*n2[2] - n1[2]*n2[1] == 0
    end
    return false
end

"""
Check if two integer vectors are orthogonal.
"""
function are_orthogonal(n1::Vector{Int}, n2::Vector{Int})
    return dot(n1, n2) == 0
end

# ============================================================================
# Main: Detect centering type from symmetry operations
# ============================================================================

function detect_centering_from_ops(ops)
    centering_vecs = Vector{Vector{Rational{Int}}}()
    D = size(first(ops).rotation, 1)
    I_mat = Matrix{Int}(I, D, D)
    
    for op in ops
        R = round.(Int, op.rotation)
        t_frac = op.translation  # already fractional in Crystalline.jl
        
        # Pure translation: R = I, t ≠ 0
        if R == I_mat && any(abs.(t_frac) .> 1e-6)
            t_rat = rationalize.(t_frac, tol=1e-6)
            push!(centering_vecs, t_rat)
        end
    end
    
    # Classify
    if isempty(centering_vecs)
        return :P, centering_vecs
    end
    
    # Check for F-centering (3 vectors: (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0))
    if length(centering_vecs) >= 3
        # Check if F-centering subset exists
        f_vecs = [[0, 1//2, 1//2], [1//2, 0, 1//2], [1//2, 1//2, 0]]
        n_f = count(v -> any(cv -> all(mod.(cv - v, 1) .== 0) || all(mod.(v - cv, 1) .== 0), centering_vecs), f_vecs)
        if n_f >= 3
            return :F, centering_vecs
        end
    end
    
    # Check for I-centering (1/2, 1/2, 1/2)
    for cv in centering_vecs
        if all(abs.(cv .- 1//2) .< 1e-6)
            return :I, centering_vecs
        end
    end
    
    # Check for C/A/B centering
    for cv in centering_vecs
        t = [rationalize(x, tol=1e-6) for x in cv]
        n_half = count(x -> x == 1//2, t)
        n_zero = count(x -> x == 0, t)
        if n_half == 2 && n_zero == 1
            if t[3] == 0
                return :C, centering_vecs
            elseif t[2] == 0
                return :B, centering_vecs
            elseif t[1] == 0
                return :A, centering_vecs
            end
        end
    end
    
    # R-centering (rhombohedral)
    for cv in centering_vecs
        if all(abs.(cv .- [2//3, 1//3, 1//3]) .< 1e-6) || all(abs.(cv .- [1//3, 2//3, 2//3]) .< 1e-6)
            return :R, centering_vecs
        end
    end
    
    return :other, centering_vecs
end

# ============================================================================
# Main: Find orthogonal mirror/glide triples
# ============================================================================

struct MirrorInfo
    R::Matrix{Int}                  # Rotation matrix
    t_frac::Vector{Rational{Int}}   # Translation (fractional)
    normal::Vector{Int}             # Mirror plane normal
    op_index::Int                   # Index in ops list
end

function find_mirrors(ops)
    mirrors = MirrorInfo[]
    seen_normals = Vector{Vector{Int}}()
    
    for (idx, op) in enumerate(ops)
        R = round.(Int, op.rotation)
        if !is_mirror_or_glide(R)
            continue
        end
        
        n = mirror_normal(R)
        if n === nothing
            continue
        end
        
        # Normalize: ensure first nonzero component is positive
        for val in n
            if val != 0
                if val < 0
                    n = -n
                end
                break
            end
        end
        
        # Skip duplicates (same normal direction)
        is_dup = false
        for sn in seen_normals
            if are_parallel(n, sn)
                is_dup = true
                break
            end
        end
        if is_dup
            continue
        end
        
        push!(seen_normals, n)
        t_rat = [rationalize(x, tol=1e-6) for x in op.translation]
        push!(mirrors, MirrorInfo(R, t_rat, n, idx))
    end
    
    return mirrors
end

"""
Find the best set of mutually orthogonal mirrors (up to 3).
Returns the set with maximum number of orthogonal mirrors.
Prefers axial normals (aligned with coordinate axes) over diagonal ones.
"""
function find_orthogonal_mirrors(mirrors::Vector{MirrorInfo})
    n = length(mirrors)
    if n == 0
        return MirrorInfo[]
    end
    
    # Score: prefer axial normals
    function axial_score(m::MirrorInfo)
        n_zeros = count(x -> x == 0, m.normal)
        return n_zeros  # Higher = more axial
    end
    
    best_triple = MirrorInfo[]
    best_score = -1
    
    # Try all triples (or pairs, or singles)
    for i in 1:n
        for j in (i+1):n
            if !are_orthogonal(mirrors[i].normal, mirrors[j].normal)
                continue
            end
            for k in (j+1):n
                if are_orthogonal(mirrors[i].normal, mirrors[k].normal) &&
                   are_orthogonal(mirrors[j].normal, mirrors[k].normal)
                    score = axial_score(mirrors[i]) + axial_score(mirrors[j]) + axial_score(mirrors[k])
                    if score > best_score || length(best_triple) < 3
                        best_triple = [mirrors[i], mirrors[j], mirrors[k]]
                        best_score = score
                    end
                end
            end
            # Also consider pair if no triple found yet
            if length(best_triple) < 2
                score = axial_score(mirrors[i]) + axial_score(mirrors[j])
                best_triple = [mirrors[i], mirrors[j]]
                best_score = score
            end
        end
        # Also consider single
        if isempty(best_triple)
            best_triple = [mirrors[i]]
            best_score = axial_score(mirrors[i])
        end
    end
    
    return best_triple
end

"""
Determine the basis transformation matrix T needed to make the mirror normals
aligned with coordinate axes.

If all normals are already axial (e.g. [1,0,0], [0,1,0], [0,0,1]),
T = I and det = 1.

If diagonal normals exist (e.g. [1,1,0] for 45° mirror), need basis change.
"""
function compute_basis_transform(mirrors::Vector{MirrorInfo})
    if isempty(mirrors)
        return Matrix{Int}(I, 3, 3), 1
    end
    
    normals = [m.normal for m in mirrors]
    D = length(normals[1])
    
    # Check if all normals are axial
    all_axial = all(n -> count(x -> x != 0, n) == 1, normals)
    
    if all_axial
        # Standard basis works
        return Matrix{Int}(I, D, D), 1
    end
    
    # Need non-standard basis
    # Build T from the normals (they become the new coordinate axes)
    if length(normals) == 3
        T = hcat(normals...)
        d = abs(round(Int, det(T)))
        return T, d
    elseif length(normals) == 2
        # Need to find a third axis orthogonal to both
        if D == 3
            n3 = cross(normals[1], normals[2])
            g = gcd(abs.(n3)...)
            if g > 0
                n3 = n3 .÷ g
            end
            T = hcat(normals..., n3)
            d = abs(round(Int, det(T)))
            return T, d
        end
    end
    
    return Matrix{Int}(I, D, D), 1
end

# ============================================================================
# Main computation loop
# ============================================================================

function compute_all_space_groups()
    results = Dict{Int, NamedTuple}()
    
    for sg in 1:230
        try
            sg_obj = spacegroup(sg, Val(3))
            ops = operations(sg_obj)
            
            # 1. Detect centering
            centering, cent_vecs = detect_centering_from_ops(ops)
            
            # 2. Find mirrors
            mirrors = find_mirrors(ops)
            
            # 3. Find orthogonal triple
            ortho_mirrors = find_orthogonal_mirrors(mirrors)
            
            # 4. Compute basis transform
            T, det_T = compute_basis_transform(ortho_mirrors)
            
            # 5. Determine DCT types needed
            n_mirrors = length(ortho_mirrors)
            has_glide = any(m -> any(x -> x != 0, m.t_frac), ortho_mirrors)
            
            # 6. Compute effective reduction factor
            mirror_reduction = 2^n_mirrors
            centering_multiplier = centering == :F ? 4 : centering == :I ? 2 : centering == :C ? 2 : 1
            theoretical_reduction = mirror_reduction * centering_multiplier
            effective_reduction = theoretical_reduction ÷ det_T
            
            # 7. Extract normals and translations
            mirror_normals = [m.normal for m in ortho_mirrors]
            mirror_translations = [m.t_frac for m in ortho_mirrors]
            mirror_Rs = [m.R for m in ortho_mirrors]
            
            # 8. Determine what crystal system (for documentation)
            results[sg] = (
                n_mirrors = n_mirrors,
                normals = mirror_normals,
                translations = mirror_translations,
                rotations = mirror_Rs,
                T = T,
                det_T = det_T,
                centering = centering,
                centering_vecs = cent_vecs,
                mirror_reduction = mirror_reduction,
                centering_multiplier = centering_multiplier,
                theoretical_reduction = theoretical_reduction,
                effective_reduction = effective_reduction,
                has_glide = has_glide,
            )
        catch e
            println("Error processing SG $sg: $e")
            results[sg] = (
                n_mirrors = 0,
                normals = Vector{Int}[],
                translations = Vector{Rational{Int}}[],
                rotations = Matrix{Int}[],
                T = Matrix{Int}(I, 3, 3),
                det_T = 1,
                centering = :P,
                centering_vecs = Vector{Rational{Int}}[],
                mirror_reduction = 1,
                centering_multiplier = 1,
                theoretical_reduction = 1,
                effective_reduction = 1,
                has_glide = false,
            )
        end
    end
    
    return results
end

"""
Generate a Julia source file containing the precomputed table.
"""
function write_table(results::Dict{Int, NamedTuple}, outpath::String)
    open(outpath, "w") do io
        println(io, "# AUTO-GENERATED by scripts/compute_mirror_table.jl")
        println(io, "# Do not edit manually. Regenerate with:")
        println(io, "#   julia --project scripts/compute_mirror_table.jl")
        println(io, "#")
        println(io, "# Mirror/Glide basis adaptation table for all 230 3D space groups.")
        println(io, "# For each space group: orthogonal mirror/glide planes, centering,")
        println(io, "# and required basis transformation.")
        println(io)
        println(io, "\"\"\"")
        println(io, "    MirrorAdaptationEntry")
        println(io)
        println(io, "Precomputed data for DCT-based CFFT for one space group.")
        println(io, "\"\"\"")
        println(io, "struct MirrorAdaptationEntry")
        println(io, "    n_mirrors::Int                               # Number of orthogonal mirror/glide planes (0-3)")
        println(io, "    normals::Vector{Vector{Int}}                 # Mirror plane normals")
        println(io, "    translations::Vector{Vector{Rational{Int}}}  # Glide translation parts (fractional)")
        println(io, "    rotations::Vector{Matrix{Int}}               # Mirror rotation matrices (3×3)")
        println(io, "    T::Matrix{Int}                               # Basis transformation matrix")
        println(io, "    det_T::Int                                   # |det(T)|, box enlargement factor")
        println(io, "    centering::Symbol                            # :P, :I, :F, :C, :A, :B, :R")
        println(io, "    centering_vecs::Vector{Vector{Rational{Int}}} # Centering translation vectors")
        println(io, "    mirror_reduction::Int                        # 2^n_mirrors")
        println(io, "    centering_multiplier::Int                    # |centering group|")
        println(io, "    theoretical_reduction::Int                   # mirror_reduction × centering_multiplier")
        println(io, "    effective_reduction::Int                     # theoretical_reduction / det_T")
        println(io, "    has_glide::Bool                              # Any glide (non-zero translation)?")
        println(io, "end")
        println(io)
        println(io, "const MIRROR_ADAPTATION_TABLE = Dict{Int, MirrorAdaptationEntry}(")
        
        for sg in 1:230
            r = results[sg]
            # Format normals
            normals_str = "[" * join(["[$(join(n, ", "))]" for n in r.normals], ", ") * "]"
            # Format translations
            trans_str = "[" * join(["[$(join(t, ", "))]" for t in r.translations], ", ") * "]"
            # Format rotations
            rot_strs = String[]
            for R in r.rotations
                rows = ["[$(join(R[i,:], " "))]" for i in 1:size(R,1)]
                push!(rot_strs, "[$(join(rows, "; "))]")
            end
            rots_str = "Matrix{Int}[" * join(rot_strs, ", ") * "]"
            # Format T
            T_rows = ["[$(join(r.T[i,:], " "))]" for i in 1:size(r.T,1)]
            T_str = "[$(join(T_rows, "; "))]"
            # Format centering vecs
            cvecs_str = "[" * join(["[$(join(v, ", "))]" for v in r.centering_vecs], ", ") * "]"
            
            println(io, "    $sg => MirrorAdaptationEntry(")
            println(io, "        $(r.n_mirrors),")
            println(io, "        $(normals_str),")
            println(io, "        $(trans_str),")
            println(io, "        $(rots_str),")
            println(io, "        $(T_str),")
            println(io, "        $(r.det_T),")
            println(io, "        :$(r.centering),")
            println(io, "        $(cvecs_str),")
            println(io, "        $(r.mirror_reduction),")
            println(io, "        $(r.centering_multiplier),")
            println(io, "        $(r.theoretical_reduction),")
            println(io, "        $(r.effective_reduction),")
            println(io, "        $(r.has_glide),")
            println(io, "    ),")
        end
        
        println(io, ")")
    end
end

"""
Print a summary report to stdout.
"""
function print_summary(results::Dict{Int, NamedTuple})
    println("\n" * "="^80)
    println("MIRROR ADAPTATION TABLE SUMMARY")
    println("="^80)
    
    # Statistics
    n3 = count(sg -> results[sg].n_mirrors == 3, 1:230)
    n2 = count(sg -> results[sg].n_mirrors == 2, 1:230)
    n1 = count(sg -> results[sg].n_mirrors == 1, 1:230)
    n0 = count(sg -> results[sg].n_mirrors == 0, 1:230)
    
    println("\nMirror count distribution:")
    println("  3 orthogonal mirrors: $n3 space groups")
    println("  2 orthogonal mirrors: $n2 space groups")
    println("  1 mirror:             $n1 space groups")
    println("  0 mirrors:            $n0 space groups")
    
    # Centering distribution
    for c in [:P, :I, :F, :C, :A, :B, :R, :other]
        nc = count(sg -> results[sg].centering == c, 1:230)
        if nc > 0
            println("  Centering $c: $nc space groups")
        end
    end
    
    # Non-trivial basis transform
    nontriv = filter(sg -> results[sg].det_T > 1, 1:230)
    println("\nNon-trivial basis transforms (|det(T)| > 1):")
    if isempty(nontriv)
        println("  None found (all axial normals)")
    else
        for sg in nontriv
            r = results[sg]
            println("  SG $sg: |det(T)| = $(r.det_T), normals = $(r.normals)")
        end
    end
    
    # Top effective reductions
    println("\nTop 20 effective reductions:")
    sorted = sort(collect(1:230), by=sg -> results[sg].effective_reduction, rev=true)
    for sg in sorted[1:min(20, end)]
        r = results[sg]
        if r.effective_reduction > 1
            println("  SG $(lpad(sg, 3)): $(rpad(r.effective_reduction, 4))× (mirrors=$(r.n_mirrors), centering=$(r.centering), det_T=$(r.det_T), glide=$(r.has_glide))")
        end
    end
    
    # Focus: groups where auto_L likely fails (has glide, no simple stride)
    println("\nGroups with glide planes (potential auto_L failures):")
    glide_groups = filter(sg -> results[sg].has_glide && results[sg].n_mirrors >= 1, 1:230)
    for sg in glide_groups
        r = results[sg]
        println("  SG $(lpad(sg, 3)): $(rpad(r.effective_reduction, 4))× ($(r.n_mirrors) mirrors, centering=$(r.centering))")
    end
end

# ============================================================================
# Run
# ============================================================================

println("Computing mirror adaptation table for all 230 3D space groups...")
results = compute_all_space_groups()

# Write output
outpath = joinpath(@__DIR__, "..", "src", "mirror_table.jl")
write_table(results, outpath)
println("Table written to: $outpath")

# Print summary
print_summary(results)
