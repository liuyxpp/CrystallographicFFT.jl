# ============================================================================
# Universal Recursive KRFFT — Per-Generator Decomposition
#
# Achieves full |G|-fold speedup by recursively applying symmetry generators:
#   1. Classify ops: diagonal-R (parity-compatible) vs non-diagonal-R (mixing)
#   2. Parity split: create 2^k sectors using diagonal ops (ALL sectors kept)
#   3. Sector orbit: use mixing ops to find equivalent sectors
#   4. Recurse on representative sectors with residual group
#   5. Butterfly recombination: standard Cooley-Tukey with orbit-aware rotation
#
# Forward:  pack rep sectors → recurse (FFT at leaf) → butterfly bottom-up → extract
# Inverse:  scatter ASU → inverse-butterfly top-down → recurse (IFFT at leaf) → unpack
# ============================================================================

# ────────────────────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────────────────────

"""
    FractalNode

Node in the recursive Cooley-Tukey decomposition tree.

- Leaf nodes: no children, have `fft_buffer` filled by direct FFT
- Inner nodes: have `children` (one per orbit representative sector),
  plus butterfly tables to combine all sectors into parent's spectral grid

Sector orbit equivalence: `n_total_sectors` sectors exist, but only
`length(children)` representative sectors need independent FFTs.
Equivalent sectors are filled by rotating the representative's FFT output.
"""
mutable struct FractalNode
    # ── Grid info ──
    subgrid_N::Vector{Int}      # spectral grid size at this level
    scale::Vector{Int}          # stride in original grid
    offset::Vector{Int}         # offset in original grid

    # ── Symmetry ──
    ops::Vector{SymOp}          # current symmetry group

    # ── Tree structure ──
    children::Vector{FractalNode}  # one per orbit representative
    parent::Union{Nothing, FractalNode}
    is_leaf::Bool

    # ── Sector info (inner nodes only) ──
    split_dims::Vector{Int}     # which dims were split (parity)
    n_total_sectors::Int        # 2^|split_dims|
    sector_parities::Vector{Vector{Int}}  # parity of each sector (D-vector)

    # Orbit mapping: for each of the n_total_sectors sectors:
    #   sector_rep_idx[s] = index into children[] for this sector's representative
    #     0 means extinct sector (centering: F=0 at those spectral positions)
    #   sector_rot[s] = rotation matrix mapping rep's FFT → sector's FFT
    #   sector_phase[s] = additional phase correction
    sector_rep_idx::Vector{Int}
    sector_rot::Vector{Matrix{Float64}}
    sector_trans::Vector{Vector{Float64}}  # translation of relating op (for phase correction)

    # ── Butterfly tables (precomputed) ──
    # For each sector s and each h in parent grid:
    #   parent[h] += twiddle[s,h] * rep_buffer[rot_map[s,h]]
    butterfly_twiddles::Vector{Vector{ComplexF64}}  # [sector][h_lin]
    butterfly_rot_maps::Vector{Vector{Int32}}       # [sector][h_lin] → index in rep buffer

    # ── Centering info (KRFFT III) ──
    # When is_centering_split=true, this node uses centering decomposition.
    # Children are centering channels with special packing (linear combinations
    # of spatial data via half-grid translations).
    # centering_pack_coeffs[child_idx][coset_k] = coefficient for coset member k
    # centering_pack_offsets[coset_k] = spatial offset for coset member k (in grid coords)
    # centering_pack_twiddle_h[child_idx] = spectral offset vector for e_A modulation
    is_centering_split::Bool
    centering_pack_coeffs::Vector{Vector{Float64}}     # [channel][coset_member]
    centering_pack_offsets::Vector{Vector{Int}}         # [coset_member] → D-vector offset
    centering_pack_twiddle_h::Vector{Vector{Float64}}   # [channel] → D-vector spectral offset

    # ── FFT buffer ──
    fft_buffer::Vector{ComplexF64}

    function FractalNode(;
        subgrid_N, scale, offset, ops,
        children=FractalNode[], parent=nothing, is_leaf=false,
        split_dims=Int[], n_total_sectors=0,
        sector_parities=Vector{Int}[],
        sector_rep_idx=Int[], sector_rot=Matrix{Float64}[], sector_trans=Vector{Float64}[],
        butterfly_twiddles=Vector{ComplexF64}[],
        butterfly_rot_maps=Vector{Int32}[],
        is_centering_split=false,
        centering_pack_coeffs=Vector{Float64}[],
        centering_pack_offsets=Vector{Int}[],
        centering_pack_twiddle_h=Vector{Float64}[],
        fft_buffer=ComplexF64[],
    )
        new(subgrid_N, scale, offset, ops,
            children, parent, is_leaf,
            split_dims, n_total_sectors, sector_parities,
            sector_rep_idx, sector_rot, sector_trans,
            butterfly_twiddles, butterfly_rot_maps,
            is_centering_split,
            centering_pack_coeffs, centering_pack_offsets, centering_pack_twiddle_h,
            fft_buffer)
    end
end

"""
    FractalCFFTPlan

Complete plan for universal recursive KRFFT.
"""
struct FractalCFFTPlan
    root::FractalNode
    fft_plans::Dict{Tuple, Any}
    spec_asu::SpectralIndexing
    extract_indices::Vector{Int32}
    output_buffer::Vector{ComplexF64}
    grid_N::Vector{Int}
    n_leaves::Int
    n_inner::Int
end

# ────────────────────────────────────────────────────────────────────────────
# Op Classification
# ────────────────────────────────────────────────────────────────────────────

"""
    classify_ops(ops, D) → (diag_ops, mixing_ops, diag_indices, mixing_indices)

Separate symmetry operations into:
- `diag_ops`: diagonal R matrices (±1 on diagonal, 0 elsewhere) — preserve per-dim parity
- `mixing_ops`: non-diagonal R matrices — mix coordinates across dimensions
"""
function classify_ops(ops::Vector{SymOp}, D::Int)
    diag_indices = Int[]
    mixing_indices = Int[]
    for (i, op) in enumerate(ops)
        R = round.(Int, op.R)
        is_diag = true
        for r in 1:D, c in 1:D
            if r != c && R[r,c] != 0
                is_diag = false; break
            end
        end
        if is_diag
            push!(diag_indices, i)
        else
            push!(mixing_indices, i)
        end
    end
    return ops[diag_indices], ops[mixing_indices], diag_indices, mixing_indices
end

"""
    find_split_dims(diag_ops, N, D) → split_dims

Find which dimensions can be split by parity: dim d is splittable if
N[d] is even AND all diagonal ops preserve parity in dim d.
"""
function find_split_dims(diag_ops::Vector{SymOp}, N, D::Int)
    split_dims = Int[]
    for d in 1:D
        N[d] % 2 != 0 && continue
        N[d] <= 1 && continue
        # Check: for all diag ops, does R[d,d] * p + t[d] preserve parity?
        #   even → even: R[d,d]*0 + t[d] must be even → t[d] even
        #   odd → odd: R[d,d]*1 + t[d] must be odd
        # For R[d,d] = ±1:
        #   t[d] must be even (integer, divisible by 2 in the N-grid)
        parity_ok = true
        for op in diag_ops
            Rdd = round(Int, op.R[d,d])
            td = round(Int, op.t[d])
            # even parity test: Rdd*0 + td = td → must be even
            if mod(td, 2) != 0
                parity_ok = false; break
            end
            # odd parity test: Rdd*1 + td → must be odd
            if mod(Rdd + td, 2) != 1
                parity_ok = false; break
            end
        end
        if parity_ok
            push!(split_dims, d)
        end
    end
    return split_dims
end

# ────────────────────────────────────────────────────────────────────────────
# Sector Orbit Detection
# ────────────────────────────────────────────────────────────────────────────

"""
    find_sector_orbits(parities, all_ops, split_dims, N, D)

Find orbit equivalences among parity sectors using ALL ops (including mixing).

Two sectors p_A and p_B are equivalent if ∃ op g such that:
  R_g * p_A + t_g ≡ p_B (mod 2) in the split dimensions

Returns:
- `orbit_rep`: orbit_rep[s] = index of representative sector for sector s
- `orbit_op`: orbit_op[s] = the SymOp mapping rep → sector s (nothing for reps)
"""
function find_sector_orbits(parities::Vector{Vector{Int}}, all_ops::Vector{SymOp},
                            split_dims::Vector{Int}, N, D::Int)
    n_sectors = length(parities)
    orbit_rep = collect(1:n_sectors)  # initially each sector is its own rep
    orbit_op = Vector{Union{Nothing, SymOp}}(fill(nothing, n_sectors))

    # Build parity → sector index lookup
    parity_to_idx = Dict{Vector{Int}, Int}()
    for (i, p) in enumerate(parities)
        parity_to_idx[p] = i
    end

    visited = falses(n_sectors)
    for i in 1:n_sectors
        visited[i] && continue
        visited[i] = true

        # Try all ops to find equivalent sectors
        for op in all_ops
            R = round.(Int, op.R)
            t = round.(Int, op.t)
            # Compute image parity: (R * p_i + t) mod 2 for split dims
            p_image = copy(parities[i])
            for d in 1:D
                s = t[d]
                for j in 1:D
                    s += R[d,j] * parities[i][j]
                end
                p_image[d] = mod(s, 2)
            end
            # Zero out non-split dims
            for d in 1:D
                if !(d in split_dims)
                    p_image[d] = 0
                end
            end

            j_idx = get(parity_to_idx, p_image, nothing)
            if j_idx !== nothing && !visited[j_idx]
                visited[j_idx] = true
                orbit_rep[j_idx] = i
                orbit_op[j_idx] = op
            end
        end
    end

    return orbit_rep, orbit_op
end

"""
    shift_ops_half_grid(ops, N, D) → shifted_ops

Apply b=1/2 grid-origin shift to symmetry ops (KRFFT V, eq. 6).

For periodic functions, sampling at any offset is valid. Choosing b=1/2
transforms: t_new = t + (I - R) · b. For a diagonal reflection R_dd = -1,
this adds 1 to t_d, making the reflection map even-parity sectors to
odd-parity sectors (cross-sector orbit equivalence). This is the key
mechanism enabling |G|-fold reduction.
"""
function shift_ops_half_grid(ops::Vector{SymOp}, N, D::Int)
    b = fill(0.5, D)
    shifted = similar(ops)
    for (i, op) in enumerate(ops)
        R = round.(Int, op.R)
        t_new = copy(op.t)
        for d in 1:D
            dt = 0.0
            for j in 1:D
                dt += (Float64(d == j) - R[d,j]) * b[j]
            end
            t_new[d] = mod(t_new[d] + dt, N[d])
        end
        t_int = round.(Int, t_new)
        if maximum(abs.(t_new .- t_int)) < 1e-10
            t_new = Float64.(t_int)
        end
        shifted[i] = SymOp(Float64.(R), t_new)
    end
    return shifted
end

# ────────────────────────────────────────────────────────────────────────────
# Tree Construction
# ────────────────────────────────────────────────────────────────────────────

"""
    build_recursive_tree(N, ops) → FractalNode

Build the universal recursive Cooley-Tukey decomposition tree.

Uses b=1/2 grid-origin shift so that diagonal symmetry operations
(reflections, inversions) create cross-sector orbit equivalences,
enabling full |G|-fold reduction.
"""
function build_recursive_tree(N::Tuple, ops::Vector{SymOp})
    D = length(N)
    root = FractalNode(
        subgrid_N=collect(N), scale=ones(Int, D), offset=zeros(Int, D), ops=ops
    )

    # BFS queue
    queue = [root]
    while !isempty(queue)
        node = pop!(queue)
        curr_N = Tuple(node.subgrid_N)

        if any(x -> x == 0, curr_N)
            node.is_leaf = true
            continue
        end

        # Step 1: Classify ops (use ORIGINAL ops for parity check)
        diag_ops, mixing_ops, _, _ = classify_ops(node.ops, D)

        # Check for non-trivial rotation ops (R ≠ I)
        has_nontrivial = any(op -> !all(op.R[i,j] == (i==j ? 1 : 0) for i in 1:D, j in 1:D), node.ops)
        if !has_nontrivial
            # No rotation ops left. Check for centering translations (R=I, t≠0 mod N)
            # IMPORTANT: After coordinate transforms, t values may equal N (full-period),
            # which is trivial (identity). Only keep genuinely non-trivial translations.
            cent_ops = SymOp[]
            for op in node.ops
                if !all(op.R[i,j] == (i==j ? 1 : 0) for i in 1:D, j in 1:D)
                    continue
                end
                t_mod = [mod(round(Int, op.t[d]), curr_N[d]) for d in 1:D]
                if any(t_mod .!= 0)
                    push!(cent_ops, SymOp(op.R, Float64.(t_mod)))
                end
            end
            if !isempty(cent_ops) && all(x -> x >= 2, curr_N)
                # Centering split (KRFFT III): decompose into multi-function channels
                _setup_centering_split!(node, cent_ops, curr_N, D)
                continue
            end
            node.is_leaf = true
            continue
        end

        # Step 2: Find splittable dims (original ops for parity preservation)
        split_dims = find_split_dims(diag_ops, curr_N, D)

        if isempty(split_dims) || all(x -> x <= 1, curr_N)
            node.is_leaf = true
            continue
        end

        node.split_dims = split_dims
        n_split = length(split_dims)
        n_sectors = 1 << n_split
        node.n_total_sectors = n_sectors

        # Step 3: Generate all sector parities
        parities = Vector{Vector{Int}}(undef, n_sectors)
        for bits in 0:n_sectors-1
            p = zeros(Int, D)
            for (k, d) in enumerate(split_dims)
                p[d] = (bits >> (k-1)) & 1
            end
            parities[bits+1] = p
        end
        node.sector_parities = parities

        # Step 4: Apply b=1/2 shift and find sector orbits
        shifted_ops = shift_ops_half_grid(node.ops, curr_N, D)
        orbit_rep, orbit_op = find_sector_orbits(parities, shifted_ops, split_dims, curr_N, D)

        # Step 5: Build children for representative sectors
        S_diag = ones(Int, D)
        for d in split_dims
            S_diag[d] = 2
        end
        child_N = [d in split_dims ? curr_N[d] ÷ 2 : curr_N[d] for d in 1:D]

        # Find unique representatives
        rep_indices = sort(unique(orbit_rep))
        rep_to_child_idx = Dict{Int, Int}()
        for (ci, ri) in enumerate(rep_indices)
            rep_to_child_idx[ri] = ci
        end

        node.sector_rep_idx = [rep_to_child_idx[orbit_rep[s]] for s in 1:n_sectors]
        node.sector_rot = Vector{Matrix{Float64}}(undef, n_sectors)
        node.sector_trans = Vector{Vector{Float64}}(undef, n_sectors)
        for s in 1:n_sectors
            op = orbit_op[s]
            if op === nothing
                node.sector_rot[s] = Matrix{Float64}(I(D))
                node.sector_trans[s] = zeros(Float64, D)
            else
                node.sector_rot[s] = round.(Int, op.R) * 1.0
                node.sector_trans[s] = Float64.(op.t)
            end
        end

        for ri in rep_indices
            parity = parities[ri]

            # Compute residual group: ops in SHIFTED system that map sector to itself
            new_ops = SymOp[]
            for (oi, op) in enumerate(node.ops)
                R = round.(Int, op.R)
                sop = shifted_ops[oi]
                t_sh = round.(Int, sop.t)

                # Image parity under shifted op
                img_p = zeros(Int, D)
                for d in 1:D
                    s = t_sh[d]
                    for j in 1:D
                        s += R[d,j] * parity[j]
                    end
                    img_p[d] = mod(s, 2)
                end
                maps_to_self = true
                for d in split_dims
                    if img_p[d] != parity[d]
                        maps_to_self = false; break
                    end
                end
                !maps_to_self && continue

                # Transform op to child coords: R_new = S⁻¹RS, t_new = S⁻¹(R*p + t_orig - p)
                # IMPORTANT: Use ORIGINAL (unshifted) translation op.t, NOT t_sh.
                # The shifted ops are used only for parity mapping above. Child ops
                # must be in the unshifted frame so each recursion level can apply
                # its own b=1/2 shift independently. Using t_sh here would cause a
                # double-shift bug for ops where (I-R)·b ≠ 0 (e.g., 3-fold rotations
                # composed with 2-fold axes in P23).
                t_orig = round.(Int, op.t)
                R_new = zeros(Float64, D, D)
                t_new = zeros(Float64, D)
                valid = true
                for i in 1:D
                    for j in 1:D
                        rv = R[i,j] * S_diag[j]
                        if rv % S_diag[i] != 0
                            valid = false; break
                        end
                        R_new[i,j] = rv ÷ S_diag[i]
                    end
                    !valid && break
                    tv = t_orig[i] - parity[i]
                    for j in 1:D
                        tv += R[i,j] * parity[j]
                    end
                    if tv % S_diag[i] != 0
                        valid = false; break
                    end
                    t_new[i] = tv ÷ S_diag[i]
                end
                if valid
                    push!(new_ops, SymOp(R_new, t_new))
                end
            end

            # Deduplicate ops
            unique_ops = _deduplicate_ops(new_ops, D)

            child = FractalNode(
                subgrid_N=copy(child_N),
                scale=node.scale .* S_diag,
                offset=node.scale .* parity .+ node.offset,
                ops=unique_ops,
                parent=node,
            )
            push!(node.children, child)
            push!(queue, child)
        end
    end

    return root
end

# ────────────────────────────────────────────────────────────────────────────
# Centering Split (KRFFT III)
# ────────────────────────────────────────────────────────────────────────────

"""
    _setup_centering_split!(node, cent_ops, curr_N, D)

Set up centering decomposition (inspired by KRFFT III, Rowicka et al. 2003).

When only centering translations remain (R=I, t≠0), perform a parity split
where centering extinction eliminates sectors with cancelling phases.

Centering creates periodicity WITHIN each stride-2 sector, not BETWEEN sectors.
Therefore we only apply EXTINCTION (skip sectors where F=0), with NO orbit
equivalence. Each alive sector gets its own independent child.
"""
function _setup_centering_split!(node::FractalNode, cent_ops::Vector{SymOp},
                                  curr_N::Tuple, D::Int)
    node.is_centering_split = true

    # Split all even dimensions
    split_dims = [d for d in 1:D if curr_N[d] >= 2 && curr_N[d] % 2 == 0]
    node.split_dims = split_dims
    n_split = length(split_dims)
    n_sectors = 1 << n_split
    node.n_total_sectors = n_sectors

    # Generate all sector parities
    parities = Vector{Vector{Int}}(undef, n_sectors)
    for bits in 0:n_sectors-1
        p = zeros(Int, D)
        for (k, d) in enumerate(split_dims)
            p[d] = (bits >> (k-1)) & 1
        end
        parities[bits+1] = p
    end
    node.sector_parities = parities

    # Build centering group (all compositions of centering translations + identity)
    cent_group = Set{Vector{Int}}()
    push!(cent_group, zeros(Int, D))
    queue_cg = [zeros(Int, D)]
    for op in cent_ops
        t = round.(Int, op.t)
        if t ∉ cent_group
            push!(cent_group, t)
            push!(queue_cg, t)
        end
    end
    while !isempty(queue_cg)
        g = pop!(queue_cg)
        for op in cent_ops
            t = round.(Int, op.t)
            h = [mod(g[d] + t[d], curr_N[d]) for d in 1:D]
            if h ∉ cent_group
                push!(cent_group, h)
                push!(queue_cg, h)
            end
        end
    end

    # Identify extinction: F(h) = 0 when Σ_{τ∈G} exp(-2πi h·τ/N) = 0
    sector_alive = trues(n_sectors)
    for s in 1:n_sectors
        p = parities[s]
        phase_sum = 0.0im
        for τ in cent_group
            phase = sum(p[d] * τ[d] / curr_N[d] for d in 1:D)
            phase_sum += cispi(-2 * phase)
        end
        if abs(phase_sum) < 0.5
            sector_alive[s] = false
        end
    end

    # Build children: one per ALIVE sector (no orbit equivalence)
    S_diag = ones(Int, D)
    for d in split_dims; S_diag[d] = 2; end
    child_N = [d in split_dims ? curr_N[d] ÷ 2 : curr_N[d] for d in 1:D]

    alive_indices = [s for s in 1:n_sectors if sector_alive[s]]
    alive_to_child = Dict{Int, Int}()
    for (ci, si) in enumerate(alive_indices)
        alive_to_child[si] = ci
    end

    # sector_rep_idx: 0 for extinct, unique child index for alive
    node.sector_rep_idx = zeros(Int, n_sectors)
    for s in 1:n_sectors
        if sector_alive[s]
            node.sector_rep_idx[s] = alive_to_child[s]
        end
    end

    # For centering butterfly: R = I, t = 0 for all sectors
    # (the twiddle handles everything)
    node.sector_rot = [Matrix{Float64}(I(D)) for _ in 1:n_sectors]
    node.sector_trans = [zeros(Float64, D) for _ in 1:n_sectors]

    # Create children: one per ALIVE sector, each is a leaf
    n_children = length(alive_indices)
    for ci in 1:n_children
        si = alive_indices[ci]
        parity = parities[si]
        child = FractalNode(
            subgrid_N=copy(child_N),
            scale=node.scale .* S_diag,
            offset=node.scale .* parity .+ node.offset,
            ops=SymOp[],  # no remaining ops
            parent=node,
            is_leaf=true,
        )
        push!(node.children, child)
    end

    # Standard stride-2 packing is correct: centering periodicity within each
    # sector means extinct sectors have zero FFT, handled by twiddle=0 in butterfly.
    node.centering_pack_coeffs = Vector{Float64}[]
    node.centering_pack_offsets = Vector{Int}[]
    node.centering_pack_twiddle_h = Vector{Float64}[]
end

function _deduplicate_ops(ops::Vector{SymOp}, D::Int)
    seen = Set{Tuple{Vector{Int}, Vector{Int}}}()
    unique_ops = SymOp[]
    for op in ops
        R_int = round.(Int, vec(op.R))
        t_int = round.(Int, op.t)
        key = (R_int, t_int)
        if !(key in seen)
            push!(seen, key)
            push!(unique_ops, op)
        end
    end
    return unique_ops
end

# ────────────────────────────────────────────────────────────────────────────
# Tree Utilities
# ────────────────────────────────────────────────────────────────────────────

"""Collect all leaf nodes."""
function collect_leaves(node::FractalNode)
    leaves = FractalNode[]
    _collect_leaves!(node, leaves)
    return leaves
end
function _collect_leaves!(node::FractalNode, leaves::Vector{FractalNode})
    if node.is_leaf
        push!(leaves, node)
    else
        for child in node.children
            _collect_leaves!(child, leaves)
        end
    end
end

"""Collect inner nodes in bottom-up order (children before parents)."""
function collect_inner_nodes_bottomup(node::FractalNode)
    nodes = FractalNode[]
    _collect_inner_bottomup!(node, nodes)
    return nodes
end
function _collect_inner_bottomup!(node::FractalNode, nodes::Vector{FractalNode})
    for child in node.children
        _collect_inner_bottomup!(child, nodes)
    end
    if !node.is_leaf
        push!(nodes, node)
    end
end

"""Tree summary statistics."""
function tree_summary(root::FractalNode)
    leaves = collect_leaves(root)
    inner = collect_inner_nodes_bottomup(root)
    max_depth = 0
    for leaf in leaves
        d = 0
        node = leaf
        while node.parent !== nothing
            d += 1
            node = node.parent
        end
        max_depth = max(max_depth, d)
    end
    (n_gp_leaves=length(leaves), n_sp_nodes=length(inner), max_depth=max_depth)
end

# ────────────────────────────────────────────────────────────────────────────
# Plan Construction
# ────────────────────────────────────────────────────────────────────────────

"""
    plan_fractal_krfft(spec_asu, ops) → FractalCFFTPlan

Build a universal recursive KRFFT plan.
"""
function plan_fractal_krfft(spec_asu::SpectralIndexing, ops::Vector{SymOp})
    N = spec_asu.N
    D = length(N)

    # Build tree
    root = build_recursive_tree(Tuple(N), ops)

    # Allocate buffers
    _allocate_buffers!(root)

    # Create FFT plans (shared by size)
    fft_plans = Dict{Tuple, Any}()
    _create_fft_plans!(fft_plans, root)

    # Precompute butterfly tables for inner nodes
    _precompute_butterfly!(root, N)

    # Build spectral ASU extraction mapping
    n_spec = length(spec_asu.points)
    extract_indices = Vector{Int32}(undef, n_spec)
    root_N = Tuple(root.subgrid_N)
    for (i, pt) in enumerate(spec_asu.points)
        raw_k = pt.idx
        lin = 1
        stride = 1
        for d in 1:D
            lin += raw_k[d] * stride
            stride *= root_N[d]
        end
        extract_indices[i] = Int32(lin)
    end

    output_buffer = zeros(ComplexF64, n_spec)
    leaves = collect_leaves(root)
    inner = collect_inner_nodes_bottomup(root)
    total_fft = sum(prod(n.subgrid_N) for n in leaves)

    @info "Universal KRFFT plan: n_spec=$n_spec, $(length(leaves)) leaves, " *
          "$(length(inner)) inner nodes, total_fft_pts=$total_fft, N=$(Tuple(N))"

    return FractalCFFTPlan(
        root, fft_plans, spec_asu, extract_indices, output_buffer,
        collect(N), length(leaves), length(inner),
    )
end

function _allocate_buffers!(node::FractalNode)
    vol = prod(node.subgrid_N)
    node.fft_buffer = zeros(ComplexF64, max(vol, 1))
    for child in node.children
        _allocate_buffers!(child)
    end
end

function _create_fft_plans!(fft_plans::Dict, node::FractalNode)
    if node.is_leaf
        dims = Tuple(node.subgrid_N)
        if !haskey(fft_plans, dims) && prod(dims) > 0
            dummy = zeros(ComplexF64, dims)
            fft_plans[dims] = plan_fft(dummy)
        end
    end
    for child in node.children
        _create_fft_plans!(fft_plans, child)
    end
end

"""
Precompute butterfly twiddle factors and rotation maps for inner nodes.

For each sector s, for each spectral point h in parent grid:
  parent[h] += twiddle_s(h) × child_rep[rot_map_s(h)]

The twiddle is: exp(-2πi Σ_d h_d * p_s_d / parent_N_d)
where parent_N is the parent's subgrid size at this recursion level.

The rot_map maps h through:
1. Apply the sector's relating rotation R_g: h' = R_g^T h
2. Reduce to child grid: h'' = h' mod child_N
3. Convert to linear index
"""
function _precompute_butterfly!(node::FractalNode, N)
    if node.is_leaf
        return
    end

    D = length(N)
    parent_N = Tuple(node.subgrid_N)
    parent_vol = prod(parent_N)
    # N_full = node.scale .* node.subgrid_N  # effective grid size

    n_sectors = node.n_total_sectors
    node.butterfly_twiddles = Vector{Vector{ComplexF64}}(undef, n_sectors)
    node.butterfly_rot_maps = Vector{Vector{Int32}}(undef, n_sectors)

    for s in 1:n_sectors
        parity = node.sector_parities[s]
        child_idx = node.sector_rep_idx[s]
        if child_idx == 0  # extinct sector (centering)
            # Fill with zeros — no contribution to parent
            node.butterfly_twiddles[s] = zeros(ComplexF64, parent_vol)
            node.butterfly_rot_maps[s] = ones(Int32, parent_vol)  # dummy (won't matter)
            continue
        end
        R_g = round.(Int, node.sector_rot[s])
        t_g = node.sector_trans[s]
        child = node.children[child_idx]
        child_N = child.subgrid_N

        # Compute offset correction δ for orbit equivalence:
        # All in NODE-LOCAL coordinates (the node's own grid [0, subgrid_N)):
        #   δ = S_diag^{-1} (R^T (p_equiv - t_g) - p_rep)
        # where p_equiv/p_rep are parity vectors (0-1 in split dims, 0 elsewhere),
        #       t_g is the orbit translation (in node's coordinate system),
        #       S_diag = 2 for split dims, 1 otherwise
        # Phase correction: exp(+2πi h·R·δ/M) where M = child_N
        #
        # Find the actual parity representative for this sector's rep
        rep_sector_idx = findfirst(i -> node.sector_rep_idx[i] == child_idx &&
                                       i == findfirst(j -> node.sector_rep_idx[j] == child_idx, 1:n_sectors),
                                   1:n_sectors)
        p_rep = node.sector_parities[rep_sector_idx]
        p_equiv = parity

        S_diag = ones(Int, D)
        for d in node.split_dims; S_diag[d] = 2; end

        # R^T * (p_equiv - t_g) in node coords
        pe_minus_t = Float64.(p_equiv) .- t_g
        Rt_pe = [sum(R_g[j,d] * pe_minus_t[j] for j in 1:D) for d in 1:D]  # R^T * (p_e - t)
        delta = round.(Int, (Rt_pe .- p_rep) ./ S_diag)
        R_delta = [sum(R_g[d,j] * delta[j] for j in 1:D) for d in 1:D]

        twiddles = Vector{ComplexF64}(undef, parent_vol)
        rot_map = Vector{Int32}(undef, parent_vol)

        h_lin = 0
        for ci in CartesianIndices(parent_N)
            h_lin += 1
            h = collect(Tuple(ci)) .- 1  # 0-based

            # Combined twiddle = parity phase × offset correction:
            #   exp(-2πi Σ h_d p_d / parent_N_d)           ← parity (Cooley-Tukey)
            # × exp(+2πi Σ h_d R_delta_d / child_N_d)      ← orbit offset correction
            # NOTE: t_g is already captured by δ, do NOT add t_g separately
            phase = 0.0
            for d in 1:D
                phase += h[d] * parity[d] / parent_N[d]
                phase -= h[d] * R_delta[d] / child_N[d]
            end
            twiddles[h_lin] = cispi(-2 * phase)

            # Rotation: h' = R_g^T * h (= R_g⁻¹ h for orthogonal R)
            # Verified numerically: Y_s(h) = Y_rep(R_g⁻¹ h) = Y_rep(R_g^T h)
            child_lin = 1
            stride = 1
            for d in 1:D
                hd_rot = 0
                for j in 1:D
                    hd_rot += R_g[j, d] * h[j]  # (R_g^T h)[d] = Σ_j R_g[j,d] * h[j]
                end
                hd_rot = mod(hd_rot, child_N[d])
                child_lin += hd_rot * stride
                stride *= child_N[d]
            end
            rot_map[h_lin] = Int32(child_lin)
        end

        node.butterfly_twiddles[s] = twiddles
        node.butterfly_rot_maps[s] = rot_map
    end

    # Recurse
    for child in node.children
        _precompute_butterfly!(child, N)
    end
end

# ────────────────────────────────────────────────────────────────────────────
# Forward Transform
# ────────────────────────────────────────────────────────────────────────────

"""
    execute_fractal_krfft!(plan, u) → spectral ASU values

Forward transform pipeline:
1. Pack leaf buffers from real-space data
2. FFT each leaf
3. Butterfly bottom-up: combine children → parent
4. Extract spectral ASU from root
"""
function execute_fractal_krfft!(plan::FractalCFFTPlan, u::AbstractArray{<:Real})
    N = Tuple(plan.grid_N)
    D = length(N)

    # Step 1+2: Pack and FFT leaves
    leaves = collect_leaves(plan.root)
    for leaf in leaves
        _pack_leaf!(leaf, u, N, D)
        dims = Tuple(leaf.subgrid_N)
        if prod(dims) > 0
            p = plan.fft_plans[dims]
            buf_nd = reshape(leaf.fft_buffer, dims)
            tmp = p * buf_nd
            copyto!(leaf.fft_buffer, vec(tmp))
        end
    end

    # Step 3: Butterfly bottom-up
    inner_nodes = collect_inner_nodes_bottomup(plan.root)
    for node in inner_nodes
        _butterfly!(node)
    end

    # Step 4: Extract spectral ASU
    root_buf = plan.root.fft_buffer
    out = plan.output_buffer
    @inbounds for i in eachindex(out)
        out[i] = root_buf[plan.extract_indices[i]]
    end

    return out
end

"""Pack a leaf's buffer from the full real-space array."""
function _pack_leaf!(leaf::FractalNode, u::AbstractArray{<:Real}, N::Tuple, D::Int)
    dims = Tuple(leaf.subgrid_N)
    scale = leaf.scale
    offset = leaf.offset

    idx = 0
    for ci in CartesianIndices(dims)
        idx += 1
        local_coord = Tuple(ci) .- 1
        gi = ntuple(d -> mod(scale[d] * local_coord[d] + offset[d], N[d]) + 1, D)
        @inbounds leaf.fft_buffer[idx] = complex(u[gi...])
    end
end

"""
Bottom-up butterfly: combine all sectors into parent's spectral grid.

parent[h] = Σ_{s=1}^{n_sectors} twiddle_s[h] × child_rep_s[rot_map_s[h]]
"""
function _butterfly!(node::FractalNode)
    if isempty(node.children)
        return
    end

    fill!(node.fft_buffer, zero(ComplexF64))

    @inbounds for s in 1:node.n_total_sectors
        child_idx = node.sector_rep_idx[s]
        child_idx == 0 && continue  # skip extinct sectors (centering)
        child_buf = node.children[child_idx].fft_buffer
        twiddles = node.butterfly_twiddles[s]
        rot_map = node.butterfly_rot_maps[s]

        for h_lin in eachindex(node.fft_buffer)
            node.fft_buffer[h_lin] += twiddles[h_lin] * child_buf[rot_map[h_lin]]
        end
    end
end

# ────────────────────────────────────────────────────────────────────────────
# Legacy aliases (keep backward compatibility)
# ────────────────────────────────────────────────────────────────────────────

const calc_asu_tree = build_recursive_tree

# ────────────────────────────────────────────────────────────────────────────
# Optimized Execution: O(N_spec) pack + sparse butterfly
# ────────────────────────────────────────────────────────────────────────────

"""
    SparseButterflyOp

Precomputed butterfly operation for one (node, sector) combination,
operating only on needed frequency indices.
Fields use ABSOLUTE indices into the unified buffer pool.
"""
struct SparseButterflyOp
    twiddles::Vector{ComplexF64}    # [n_entries] twiddle factors
    child_pool_idx::Vector{Int32}  # [n_entries] absolute pool index for child read
    parent_pool_idx::Vector{Int32} # [n_entries] absolute pool index for parent write
end

"""
    ButterflyGroup

All butterfly operations for one inner node + the zero-fill for its parent buffer.
"""
struct ButterflyGroup
    parent_zero_indices::Vector{Int32}  # pool indices that must be zeroed
    ops::Vector{SparseButterflyOp}      # one per active sector
end

"""
    LeafFFTInfo

FFT info for a single leaf, storing its pool offset and dimensions.
"""
struct LeafFFTInfo
    pool_offset::Int    # 0-based offset in buffer_pool
    dims::Tuple         # subgrid dimensions
end

"""
    OptimizedFractalPlan

Optimized plan that achieves O(N_spec) pack + sparse butterfly.
All buffers are flattened into a single contiguous pool.
"""
struct OptimizedFractalPlan
    # ── Buffer pool ──
    buffer_pool::Vector{ComplexF64}

    # ── Pack (parallel src/dst arrays) ──
    pack_src::Vector{Int32}    # [total_fft_pts] source index in flat u
    pack_dst::Vector{Int32}    # [total_fft_pts] destination pool index (1-based)
    total_fft_pts::Int

    # ── FFT ──
    leaf_fft_infos::Vector{LeafFFTInfo}
    fft_plans::Dict{Tuple, Any}

    # ── Butterfly (sparse, bottom-up) ──
    butterfly_groups::Vector{ButterflyGroup}

    # ── Extract ──
    extract_pool_idx::Vector{Int32}  # pool indices for ASU extraction
    output_buffer::Vector{ComplexF64}

    # ── Metadata ──
    grid_N::Vector{Int}
    n_spec::Int
end

"""
    plan_fractal_krfft_v2(spec_asu, ops) → OptimizedFractalPlan

Build an optimized fractal KRFFT plan with O(N_spec) sparse butterfly.
"""
function plan_fractal_krfft_v2(spec_asu::SpectralIndexing, ops::Vector{SymOp})
    N = spec_asu.N
    D = length(N)

    # Build tree (reuse existing logic)
    root = build_recursive_tree(Tuple(N), ops)
    _allocate_buffers!(root)
    _precompute_butterfly!(root, N)

    # Create FFT plans
    fft_plans = Dict{Tuple, Any}()
    _create_fft_plans!(fft_plans, root)

    # ── Step 1: Flatten all buffers into a contiguous pool ──
    all_nodes = _collect_all_nodes(root)
    node_offsets = Dict{UInt, Int}()
    total_pool_size = 0
    for node in all_nodes
        node_offsets[objectid(node)] = total_pool_size
        total_pool_size += max(prod(node.subgrid_N), 1)
    end
    buffer_pool = zeros(ComplexF64, total_pool_size)

    # ── Step 2: Build pack gather table (pack_src, pack_dst) ──
    leaves = collect_leaves(root)
    total_fft_pts = sum(prod(l.subgrid_N) for l in leaves)
    pack_src = Vector{Int32}(undef, total_fft_pts)
    pack_dst = Vector{Int32}(undef, total_fft_pts)
    leaf_fft_infos = LeafFFTInfo[]

    pack_idx = 0
    N_tuple = Tuple(N)
    li = LinearIndices(N_tuple)
    for leaf in leaves
        off = node_offsets[objectid(leaf)]
        dims = Tuple(leaf.subgrid_N)
        push!(leaf_fft_infos, LeafFFTInfo(off, dims))

        for ci in CartesianIndices(dims)
            pack_idx += 1
            local_coord = Tuple(ci) .- 1
            gi = ntuple(d -> mod(leaf.scale[d] * local_coord[d] + leaf.offset[d],
                                 N[d]) + 1, D)
            pack_src[pack_idx] = Int32(li[gi...])
            pack_dst[pack_idx] = Int32(off + LinearIndices(dims)[ci])
        end
    end

    # Create FFT plans (out-of-place, shared by size)
    fft_plans = Dict{Tuple, Any}()
    for info in leaf_fft_infos
        dims = info.dims
        if prod(dims) > 0 && !haskey(fft_plans, dims)
            dummy = zeros(ComplexF64, dims)
            fft_plans[dims] = plan_fft(dummy)
        end
    end

    # ── Step 3: Compute needed indices top-down (sparse butterfly) ──
    n_spec = length(spec_asu.points)
    root_N = Tuple(root.subgrid_N)

    asu_lin = Vector{Int32}(undef, n_spec)
    for (i, pt) in enumerate(spec_asu.points)
        raw_k = pt.idx
        lin = 1; stride = 1
        for d in 1:D
            lin += raw_k[d] * stride
            stride *= root_N[d]
        end
        asu_lin[i] = Int32(lin)
    end

    needed_map = Dict{UInt, Vector{Int32}}()
    _compute_needed_sets!(root, asu_lin, needed_map)

    # ── Step 4: Build sparse butterfly schedule ──
    inner_nodes = collect_inner_nodes_bottomup(root)
    butterfly_groups = ButterflyGroup[]

    n_sparse = 0
    n_full = 0

    for node in inner_nodes
        needed = get(needed_map, objectid(node), Int32[])
        parent_off = node_offsets[objectid(node)]
        parent_vol = prod(node.subgrid_N)
        is_sparse = length(needed) < parent_vol

        if is_sparse
            n_sparse += 1
        else
            n_full += 1
        end

        # Build zero-fill index list
        if is_sparse
            zero_indices = Int32[parent_off + h for h in needed]
        else
            zero_indices = collect(Int32, (parent_off + 1):(parent_off + parent_vol))
        end

        # Build ops for each active sector
        ops_for_node = SparseButterflyOp[]
        active_h = is_sparse ? needed : collect(Int32, 1:parent_vol)

        for s in 1:node.n_total_sectors
            child_idx = node.sector_rep_idx[s]
            child_idx == 0 && continue
            child = node.children[child_idx]
            child_off = node_offsets[objectid(child)]
            tw_full = node.butterfly_twiddles[s]
            rm_full = node.butterfly_rot_maps[s]

            n_h = length(active_h)
            tw = Vector{ComplexF64}(undef, n_h)
            ci = Vector{Int32}(undef, n_h)
            pi = Vector{Int32}(undef, n_h)

            @inbounds for (j, h) in enumerate(active_h)
                tw[j] = tw_full[h]
                ci[j] = Int32(child_off + rm_full[h])
                pi[j] = Int32(parent_off + h)
            end

            push!(ops_for_node, SparseButterflyOp(tw, ci, pi))
        end

        push!(butterfly_groups, ButterflyGroup(zero_indices, ops_for_node))
    end

    # ── Step 5: Extract indices (absolute pool positions) ──
    root_off = node_offsets[objectid(root)]
    extract_pool_idx = Int32[root_off + h for h in asu_lin]
    output_buffer = zeros(ComplexF64, n_spec)

    @info "Optimized KRFFT v2: n_spec=$n_spec, " *
          "$(length(leaves)) leaves, $(length(inner_nodes)) inner " *
          "($(n_sparse) sparse + $(n_full) full), " *
          "pool=$total_pool_size, fft_pts=$total_fft_pts"

    return OptimizedFractalPlan(
        buffer_pool,
        pack_src, pack_dst, total_fft_pts,
        leaf_fft_infos, fft_plans,
        butterfly_groups,
        extract_pool_idx, output_buffer,
        collect(N), n_spec)
end

"""Top-down computation of needed frequency indices for each node."""
function _compute_needed_sets!(node::FractalNode, needed::Vector{Int32},
                                needed_map::Dict{UInt, Vector{Int32}})
    parent_vol = prod(node.subgrid_N)

    if length(needed) >= parent_vol
        needed_map[objectid(node)] = collect(Int32, 1:parent_vol)
    else
        needed_map[objectid(node)] = needed
    end

    if node.is_leaf
        return
    end

    # For each representative child, collect which entries are needed
    child_needed = Dict{Int, Set{Int32}}()
    for s in 1:node.n_total_sectors
        child_idx = node.sector_rep_idx[s]
        child_idx == 0 && continue
        if !haskey(child_needed, child_idx)
            child_needed[child_idx] = Set{Int32}()
        end
        rm = node.butterfly_rot_maps[s]
        actual_needed = length(needed) >= parent_vol ?
                        (1:parent_vol) : needed
        for h in actual_needed
            push!(child_needed[child_idx], rm[h])
        end
    end

    for (ci, child) in enumerate(node.children)
        cn = get(child_needed, ci, Set{Int32}())
        sorted_cn = sort!(collect(cn))
        _compute_needed_sets!(child, sorted_cn, needed_map)
    end
end

"""Collect all nodes in the tree (BFS order)."""
function _collect_all_nodes(root::FractalNode)
    nodes = FractalNode[]
    queue = [root]
    while !isempty(queue)
        node = popfirst!(queue)
        push!(nodes, node)
        for child in node.children
            push!(queue, child)
        end
    end
    return nodes
end

"""
    execute_fractal_krfft_v2!(plan, u) → spectral ASU values

Optimized forward transform with O(N_spec) pack + sparse butterfly.
"""
function execute_fractal_krfft_v2!(plan::OptimizedFractalPlan, u::AbstractArray{<:Real})
    pool = plan.buffer_pool
    u_flat = vec(u)

    # Step 1: Pack all leaves via gather table (O(N_spec))
    ps = plan.pack_src
    pd = plan.pack_dst
    @inbounds @simd for i in eachindex(ps)
        pool[pd[i]] = complex(u_flat[ps[i]])
    end

    # Step 2: FFT all leaves
    for info in plan.leaf_fft_infos
        dims = info.dims
        vol = prod(dims)
        vol == 0 && continue
        off = info.pool_offset
        fp = plan.fft_plans[dims]
        buf = reshape(@view(pool[off+1:off+vol]), dims)
        tmp = fp * buf
        copyto!(@view(pool[off+1:off+vol]), vec(tmp))
    end

    # Step 3: Sparse butterfly (bottom-up)
    for group in plan.butterfly_groups
        # Zero-fill only needed positions
        zi = group.parent_zero_indices
        @inbounds @simd for i in eachindex(zi)
            pool[zi[i]] = zero(ComplexF64)
        end

        # Accumulate contributions from all sectors
        for op in group.ops
            tw = op.twiddles
            ci = op.child_pool_idx
            pi = op.parent_pool_idx
            @inbounds @simd for j in eachindex(tw)
                pool[pi[j]] += tw[j] * pool[ci[j]]
            end
        end
    end

    # Step 4: Extract spectral ASU
    out = plan.output_buffer
    ei = plan.extract_pool_idx
    @inbounds @simd for i in eachindex(out)
        out[i] = pool[ei[i]]
    end

    return out
end

