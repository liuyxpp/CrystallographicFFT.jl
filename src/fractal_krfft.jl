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
        fft_buffer=ComplexF64[],
    )
        new(subgrid_N, scale, offset, ops,
            children, parent, is_leaf,
            split_dims, n_total_sectors, sector_parities,
            sector_rep_idx, sector_rot, sector_trans,
            butterfly_twiddles, butterfly_rot_maps,
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

        # Skip if only identity op
        has_nontrivial = any(op -> !all(op.R[i,j] == (i==j ? 1 : 0) for i in 1:D, j in 1:D), node.ops)
        if !has_nontrivial
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

                # Transform op to child coords: R_new = S⁻¹RS, t_new = S⁻¹(R*p + t_sh - p)
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
                    tv = t_sh[i] - parity[i]
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
        R_g = round.(Int, node.sector_rot[s])
        t_g = node.sector_trans[s]
        child_idx = node.sector_rep_idx[s]
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
