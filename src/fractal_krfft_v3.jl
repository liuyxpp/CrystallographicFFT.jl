# ═══════════════════════════════════════════════════════════════════════════
#  V3: Pull-based ASU-only butterfly (O(N_spec × levels) complexity)
#
#  Key insight (KRFFT V paper, eq. 13): evaluate F(h) only for h ∈ ASU,
#  using pre-computed GP FFTs as lookup tables. Each node only stores its
#  spectral ASU entries; non-ASU entries are derived via symmetry on-the-fly.
# ═══════════════════════════════════════════════════════════════════════════

using FFTW: plan_fft

"""
    NodeASUInfo

Per-node spectral ASU information. Only ASU entries are stored in the pool.
For non-ASU index k: F(k) = sym_phase[k] × F_ASU[asu_rank[sym_map[k]]]
"""
struct NodeASUInfo
    asu_indices::Vector{Int32}       # sorted linear indices of ASU entries (1-based, in subgrid)
    asu_rank::Vector{Int32}          # [vol] → position in asu_indices (0 if not ASU rep)
    sym_map::Vector{Int32}           # [vol] → ASU representative's linear index (1-based)
    sym_phase::Vector{ComplexF64}    # [vol] → phase for symmetry mapping
    pool_offset::Int                 # offset in the ASU-only pool
end

"""
    compute_spectral_asu(subgrid_N, ops) → (asu_indices, asu_rank, sym_map, sym_phase)

Compute spectral ASU for a subgrid under its residual symmetry group.

Spectral symmetry action: h → R_g^T h (mod subgrid_N)
Phase relation: F(R_g^T h) = exp(+2πi h·t_g/N) × F(h)  [from S_g*F = F]
"""
function compute_spectral_asu(subgrid_N::Vector{Int}, ops::Vector{SymOp})
    D = length(subgrid_N)
    vol = prod(subgrid_N)
    Nt = Tuple(subgrid_N)

    # Output arrays
    visited = falses(vol)
    sym_map = zeros(Int32, vol)
    sym_phase = ones(ComplexF64, vol)
    asu_list = Int32[]

    li = LinearIndices(Nt)
    ci_all = CartesianIndices(Nt)

    for k_lin in 1:vol
        visited[k_lin] && continue

        # k_lin is a new orbit representative (smallest linear index)
        push!(asu_list, Int32(k_lin))
        sym_map[k_lin] = Int32(k_lin)
        sym_phase[k_lin] = one(ComplexF64)
        visited[k_lin] = true

        k = collect(Tuple(ci_all[k_lin])) .- 1  # 0-based

        for op in ops
            R = op.R
            t = op.t

            # h' = R^T h mod N
            h_rot = zeros(Int, D)
            for d in 1:D
                for j in 1:D
                    h_rot[d] += R[j, d] * k[j]
                end
                h_rot[d] = mod(h_rot[d], subgrid_N[d])
            end

            k_rot_lin = li[(h_rot .+ 1)...]

            if !visited[k_rot_lin]
                visited[k_rot_lin] = true
                sym_map[k_rot_lin] = Int32(k_lin)

                # Phase: F(R^T h) = exp(+2πi h·t/N) × F(h)
                ph = 0.0
                for d in 1:D
                    ph += k[d] * t[d] / subgrid_N[d]
                end
                sym_phase[k_rot_lin] = cispi(2 * ph)
            end
        end
    end

    # Build rank table
    asu_rank = zeros(Int32, vol)
    for (rank, idx) in enumerate(asu_list)
        asu_rank[idx] = Int32(rank)
    end

    return asu_list, asu_rank, sym_map, sym_phase
end


"""
    ASUButterflyGroup

Butterfly for one inner node. Computes parent ASU entries from children's ASU.
"""
struct ASUButterflyGroup
    parent_pool_start::Int        # pool offset for this node's ASU
    parent_asu_size::Int          # number of ASU entries

    # Per alive sector: arrays of length parent_asu_size
    sector_twiddles::Vector{Vector{ComplexF64}}   # combined twiddle × child sym_phase
    sector_child_pool_idx::Vector{Vector{Int32}}  # absolute pool index into child ASU
end

"""
    FusedLeafDFT

Fused pack+DFT+extract for one leaf: precomputed kernel that directly maps
u[] values to ASU pool entries without any FFTW call.

For each ASU output a:
    pool[pool_offset + a] = Σ_n dft_kernel[a, n] × u_flat[u_src[n]]
"""
struct FusedLeafDFT
    pool_offset::Int                 # where ASU entries go in pool
    u_src::Vector{Int32}             # [vol] source indices in u_flat
    dft_kernel::Matrix{ComplexF64}   # [n_asu × vol] fused DFT kernel
end

"""
    ASUOnlyPlan

V3 plan: pull-based ASU-only butterfly, O(N_spec × levels) complexity.
Uses fused DFT for small leaves, per-leaf FFTW for large leaves.
"""
struct ASUOnlyPlan
    # ── Compact ASU pool ──
    asu_pool::Vector{ComplexF64}
    total_pool_size::Int

    # ── Size-1 leaves: batched vectorized gather ──
    trivial_u_idx::Vector{Int32}       # source index in u_flat
    trivial_pool_idx::Vector{Int32}    # destination in asu_pool

    # ── Small leaves (vol > 1, vol ≤ threshold): fused DFT ──
    fused_leaves::Vector{FusedLeafDFT}

    # ── Large leaves (vol > threshold): per-leaf FFTW ──
    large_leaf_pack_src::Vector{Vector{Int32}}
    large_leaf_asu_indices::Vector{Vector{Int32}}
    large_leaf_pool_offsets::Vector{Int}
    large_leaf_dims::Vector{Tuple}
    fft_plans::Dict{Tuple, Any}
    fft_temps::Dict{Tuple, Array{ComplexF64}}

    # ── ASU butterfly schedule (bottom-up, inner nodes only) ──
    butterfly_schedule::Vector{ASUButterflyGroup}

    # ── Extract (root ASU → spectral ASU output) ──
    extract_pool_idx::Vector{Int32}
    extract_phase::Vector{ComplexF64}
    output_buffer::Vector{ComplexF64}

    # ── Metadata ──
    grid_N::Vector{Int}
    n_spec::Int
end

# DFT threshold: leaves with vol ≤ this use inline DFT instead of FFTW
const FUSED_DFT_THRESHOLD = 64

"""
    _build_dft_kernel(dims, asu_indices) → Matrix{ComplexF64}

Precompute the DFT kernel for ASU entries only.
    kernel[a, n] = exp(-2πi Σ_d k_d × n_d / N_d)
where k = asu_indices[a] (converted to multi-index) and n = spatial index.
"""
function _build_dft_kernel(dims::Tuple, asu_indices::Vector{Int32})
    D = length(dims)
    vol = prod(dims)
    n_asu = length(asu_indices)

    kernel = Matrix{ComplexF64}(undef, n_asu, vol)
    ci_all = CartesianIndices(dims)

    for (a, k_lin) in enumerate(asu_indices)
        k = collect(Tuple(ci_all[k_lin])) .- 1  # 0-based frequency

        for n_lin in 1:vol
            n = collect(Tuple(ci_all[n_lin])) .- 1  # 0-based spatial
            phase = 0.0
            for d in 1:D
                phase += k[d] * n[d] / dims[d]
            end
            kernel[a, n_lin] = cispi(-2 * phase)
        end
    end

    return kernel
end


"""
    plan_fractal_krfft_v3(spec_asu, ops) → ASUOnlyPlan

Build pull-based ASU-only plan.
"""
function plan_fractal_krfft_v3(spec_asu::SpectralIndexing, ops::Vector{SymOp})
    N = spec_asu.N
    D = length(N)

    # Build tree
    root = build_recursive_tree(Tuple(N), ops)
    _allocate_buffers!(root)
    _precompute_butterfly!(root, N)

    # ── Step 1: Compute spectral ASU for each node & assign pool offsets ──
    all_nodes = _collect_all_nodes(root)
    node_asu_infos = Dict{UInt, NodeASUInfo}()

    total_pool = 0
    for node in all_nodes
        asu_indices, asu_rank, sym_map_arr, sym_phase_arr =
            compute_spectral_asu(node.subgrid_N, node.ops)
        info = NodeASUInfo(asu_indices, asu_rank, sym_map_arr, sym_phase_arr, total_pool)
        node_asu_infos[objectid(node)] = info
        total_pool += length(asu_indices)
    end

    asu_pool = zeros(ComplexF64, total_pool)

    # ── Step 2: Build per-leaf data (trivial / fused DFT / FFTW) ──
    leaves = collect_leaves(root)
    n_leaves = length(leaves)
    N_tuple = Tuple(N)
    li_u = LinearIndices(N_tuple)

    trivial_u = Int32[]
    trivial_pool = Int32[]
    fused_leaves = FusedLeafDFT[]
    large_src = Vector{Int32}[]
    large_asu = Vector{Int32}[]
    large_off = Int[]
    large_dims = Tuple[]
    fft_plans = Dict{Tuple, Any}()
    fft_temps = Dict{Tuple, Array{ComplexF64}}()
    total_pack = 0

    for leaf in leaves
        dims = Tuple(leaf.subgrid_N)
        vol = prod(dims)
        leaf_info = node_asu_infos[objectid(leaf)]

        if vol == 1
            # Trivial: batch gather
            gi = ntuple(d -> mod(leaf.offset[d], N[d]) + 1, D)
            push!(trivial_u, Int32(li_u[gi...]))
            push!(trivial_pool, Int32(leaf_info.pool_offset + 1))

        elseif vol <= FUSED_DFT_THRESHOLD
            # Small: fused pack+DFT+extract (no FFTW)
            src = Vector{Int32}(undef, vol)
            for ci in CartesianIndices(dims)
                local_coord = Tuple(ci) .- 1
                gi = ntuple(d -> mod(leaf.scale[d] * local_coord[d] + leaf.offset[d],
                                     N[d]) + 1, D)
                src[LinearIndices(dims)[ci]] = Int32(li_u[gi...])
            end
            kernel = _build_dft_kernel(dims, leaf_info.asu_indices)
            push!(fused_leaves, FusedLeafDFT(leaf_info.pool_offset, src, kernel))

        else
            # Large: per-leaf FFTW
            src = Vector{Int32}(undef, vol)
            for ci in CartesianIndices(dims)
                local_coord = Tuple(ci) .- 1
                gi = ntuple(d -> mod(leaf.scale[d] * local_coord[d] + leaf.offset[d],
                                     N[d]) + 1, D)
                src[LinearIndices(dims)[ci]] = Int32(li_u[gi...])
            end
            push!(large_src, src)
            push!(large_asu, leaf_info.asu_indices)
            push!(large_off, leaf_info.pool_offset)
            push!(large_dims, dims)
            if !haskey(fft_plans, dims)
                fft_plans[dims] = plan_fft(zeros(ComplexF64, dims))
                fft_temps[dims] = zeros(ComplexF64, dims)
            end
        end
        total_pack += vol
    end

    n_trivial = length(trivial_u)
    n_fused = length(fused_leaves)
    n_large = length(large_src)

    # ── Step 3: Build ASU butterfly schedule (bottom-up) ──
    inner_nodes = collect_inner_nodes_bottomup(root)
    butterfly_schedule = ASUButterflyGroup[]
    n_bfly = 0

    for node in inner_nodes
        pinfo = node_asu_infos[objectid(node)]
        p_asu = pinfo.asu_indices
        p_size = length(p_asu)

        s_tw = Vector{ComplexF64}[]
        s_ci = Vector{Int32}[]

        for s in 1:node.n_total_sectors
            child_idx = node.sector_rep_idx[s]
            child_idx == 0 && continue

            child = node.children[child_idx]
            cinfo = node_asu_infos[objectid(child)]

            tw_full = node.butterfly_twiddles[s]
            rm_full = node.butterfly_rot_maps[s]

            tw_vec = Vector{ComplexF64}(undef, p_size)
            ci_vec = Vector{Int32}(undef, p_size)

            @inbounds for (j, h_lin) in enumerate(p_asu)
                tw_val = tw_full[h_lin]
                k_child = rm_full[h_lin]
                k_rep = cinfo.sym_map[k_child]
                phase = cinfo.sym_phase[k_child]
                rank = cinfo.asu_rank[k_rep]
                tw_vec[j] = tw_val * phase
                ci_vec[j] = Int32(cinfo.pool_offset + rank)
            end

            n_bfly += p_size
            push!(s_tw, tw_vec)
            push!(s_ci, ci_vec)
        end

        push!(butterfly_schedule, ASUButterflyGroup(
            pinfo.pool_offset, p_size, s_tw, s_ci))
    end

    # ── Step 4: Build extract table ──
    rinfo = node_asu_infos[objectid(root)]
    n_spec = length(spec_asu.points)
    ext_idx = Vector{Int32}(undef, n_spec)
    ext_phase = Vector{ComplexF64}(undef, n_spec)

    root_N = Tuple(root.subgrid_N)
    for (i, pt) in enumerate(spec_asu.points)
        h_lin = 1; stride = 1
        for d in 1:D
            h_lin += pt.idx[d] * stride
            stride *= root_N[d]
        end
        k_rep = rinfo.sym_map[h_lin]
        ext_phase[i] = rinfo.sym_phase[h_lin]
        ext_idx[i] = Int32(rinfo.pool_offset + rinfo.asu_rank[k_rep])
    end

    fused_ops = sum(size(fl.dft_kernel, 1) * size(fl.dft_kernel, 2) for fl in fused_leaves; init=0)
    @info "KRFFT v3 (ASU-only): n_spec=$n_spec, " *
          "pool=$total_pool ($(round(total_pool/prod(N)*100, digits=1))% of N³), " *
          "$(n_leaves) leaves ($n_trivial trivial + $n_fused fused + $n_large FFTW), " *
          "pack=$total_pack, fused_ops=$fused_ops, bfly=$n_bfly"

    return ASUOnlyPlan(
        asu_pool, total_pool,
        trivial_u, trivial_pool,
        fused_leaves,
        large_src, large_asu, large_off, large_dims,
        fft_plans, fft_temps,
        butterfly_schedule,
        ext_idx, ext_phase,
        zeros(ComplexF64, n_spec),
        collect(N), n_spec)
end


"""
    execute_fractal_krfft_v3!(plan, u) → Vector{ComplexF64}

Execute pull-based ASU-only KRFFT.
"""
function execute_fractal_krfft_v3!(plan::ASUOnlyPlan, u::AbstractArray{<:Real})
    pool = plan.asu_pool
    u_flat = vec(u)

    # Step 1a: Batch gather for trivial (size-1) leaves — single SIMD loop
    tu = plan.trivial_u_idx
    tp = plan.trivial_pool_idx
    @inbounds @simd for i in eachindex(tu)
        pool[tp[i]] = complex(u_flat[tu[i]])
    end

    # Step 1b: Fused pack+DFT+extract for small leaves (no FFTW)
    @inbounds for fl in plan.fused_leaves
        poff = fl.pool_offset
        src = fl.u_src
        K = fl.dft_kernel
        n_asu = size(K, 1)
        vol = size(K, 2)

        for a in 1:n_asu
            val = zero(ComplexF64)
            for n in 1:vol
                val += K[a, n] * u_flat[src[n]]
            end
            pool[poff + a] = val
        end
    end

    # Step 1c: Per-leaf FFTW for large leaves
    for li in eachindex(plan.large_leaf_pack_src)
        dims = plan.large_leaf_dims[li]
        src = plan.large_leaf_pack_src[li]
        asu = plan.large_leaf_asu_indices[li]
        poff = plan.large_leaf_pool_offsets[li]

        tmp = plan.fft_temps[dims]  # pre-allocated small buffer (L1 cache)
        @inbounds for i in eachindex(src)
            tmp[i] = complex(u_flat[src[i]])
        end
        result = plan.fft_plans[dims] * tmp
        @inbounds for (rank, idx) in enumerate(asu)
            pool[poff + rank] = result[idx]
        end
    end

    # Step 2: ASU-only butterfly (bottom-up)
    for group in plan.butterfly_schedule
        ps = group.parent_pool_start
        sz = group.parent_asu_size

        @inbounds @simd for j in 1:sz
            pool[ps + j] = zero(ComplexF64)
        end

        for (si, tw) in enumerate(group.sector_twiddles)
            ci = group.sector_child_pool_idx[si]
            @inbounds @simd for j in 1:sz
                pool[ps + j] += tw[j] * pool[ci[j]]
            end
        end
    end

    # Step 3: Extract
    out = plan.output_buffer
    ei = plan.extract_pool_idx
    ep = plan.extract_phase
    @inbounds @simd for i in eachindex(out)
        out[i] = ep[i] * pool[ei[i]]
    end

    return out
end
