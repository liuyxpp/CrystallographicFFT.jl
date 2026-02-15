# ============================================================================
# G0 ASU Backward Transform — Factored Per-M2 Pipeline
#
# Pipeline: F_spec → inv A8 → g0_reps → fused expansion+butterfly → 4×FFT_out
#           → IFFT → unpack → sym fill → u[N³]
#
# The fused expansion+butterfly step replaces the old 4-step process
# (G_rem expand → octant read → 3-stage butterfly → collect) with a single
# CSR-indexed loop over M2³ positions using closed-form butterfly weights.
# This reduces bwd/fwd time ratio from ~2.7× to ~1.25×.
# ============================================================================

using LinearAlgebra: mul!

# ────────────────────────────────────────────────────────────────────
# Precomputed tables
# ────────────────────────────────────────────────────────────────────

"""
    InvA8GatherEntry

One weighted gather from F_spec for the inverse A8 step.
"""
struct InvA8GatherEntry
    weight::ComplexF64
    spec_idx::Int32
end

const _ZERO_INV_A8 = InvA8GatherEntry(zero(ComplexF64), Int32(1))

"""
    A8InvBlock

One block of the block-diagonal A8 inverse.
Stores the local inverse matrix and the global indices it maps.
"""
struct A8InvBlock
    spec_idxs::Vector{Int}        # which F_spec entries form this block
    rep_idxs::Vector{Int}         # which g0 orbit-rep entries this block solves for
    inv_matrix::Matrix{ComplexF64} # n×n inverse of the block's A sub-matrix
end

"""
    G0ExpansionEntry

Maps an orbit representative to another M³ position via G_rem symmetry.
"""
struct G0ExpansionEntry
    rep_compact::Int32            # source: orbit-rep compact index
    target_lin::Int32             # target: 1-based linear index into M³
    phase::ComplexF64             # G0(target) = phase × g0_rep[rep_compact]
end

"""
    OctExpEntry

One expansion entry mapped to its M2³ position and octant.
Stores the octant index (0-7), orbit-rep compact index, and phase.
Compact layout (21 bytes) for L2-cache-friendly CSR scan.
"""
struct OctExpEntry
    octant::Int8       # 0-7: xi + 2*yi + 4*zi
    rep_compact::Int32
    phase::ComplexF64
end

"""
    PerM2Table

CSR-indexed table mapping M2³ positions to their expansion entries.
For each M2 position, stores the octant entries that contribute
to the inverse butterfly at that position.
"""
struct PerM2Table
    row_ptr::Vector{Int32}          # length = M2_vol + 1
    entries::Vector{OctExpEntry}
end

# Inverse butterfly sign factors for each sub-FFT output.
# Indexed by octant+1 where octant = xi + 2*yi + 4*zi.
# oct:  0:(000) 1:(100) 2:(010) 3:(110) 4:(001) 5:(101) 6:(011) 7:(111)
# F000: always +1 (no table needed)
# F001: (-1)^zi
const _SIGN_F001 = Int8[ 1, 1, 1, 1,-1,-1,-1,-1]
# F110: (-1)^(xi⊕yi)
const _SIGN_F110 = Int8[ 1,-1,-1, 1, 1,-1,-1, 1]
# F111: (-1)^(xi⊕yi⊕zi)
const _SIGN_F111 = Int8[ 1,-1,-1, 1,-1, 1, 1,-1]

"""
    G0ASUBackwardPlan

Factored backward plan: inv A8 → fused expansion+butterfly → IFFT.
"""
struct G0ASUBackwardPlan
    # Step 1: Inverse A8 — flattened per-rep gather table
    inv_a8_table::Vector{InvA8GatherEntry}  # 8 × n_reps entries (zero-padded)
    inv_a8_counts::Vector{Int8}             # actual # of nonzero entries per rep

    # Step 2: Fused expansion + butterfly (per-M2 CSR table)
    expansion::Vector{G0ExpansionEntry}   # kept for reference/debugging
    per_m2::PerM2Table                    # CSR table with pre-multiplied weights
    tw_x::Vector{ComplexF64}
    tw_y::Vector{ComplexF64}
    tw_z::Vector{ComplexF64}

    # Step 3: IFFT
    ifft_plan::Any
    fft_bufs::Vector{Array{ComplexF64,3}}    # 4 FFT output buffers (F000,F001,F110,F111)
    ifft_out::Vector{Array{ComplexF64,3}}    # 4 IFFT output buffers

    # Step 4: Symmetry fill
    unfilled_map::Vector{Tuple{Int,Int}}

    # Intermediate storage
    g0_reps::Vector{ComplexF64}              # orbit-rep values

    # Dimensions
    grid_N::Vector{Int}
    subgrid_dims::Vector{Int}   # M = N/2
    sub_sub_dims::Vector{Int}   # M2 = N/4
    n_spec::Int
    n_reps::Int
end

"""
    plan_krfft_g0asu_backward(spec_asu, ops_shifted)

Create step-inverse backward plan by precomputing:
1. A8 block-diagonal inverse (blocks ≤ 8×8)
2. G_rem orbit expansion table
3. Butterfly twiddles (same as forward)
4. IFFT plans and symmetry fill map
"""
function plan_krfft_g0asu_backward(spec_asu::SpectralIndexing, ops_shifted::Vector{<:SymOp})
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "G0 ASU backward currently supports 3D only"
    @assert has_cubic_p3c_symmetry(ops_shifted) "G0 ASU backward requires cubic symmetry (P3c body-diagonal 3-fold rotation). " *
        "Non-cubic groups (tetragonal, orthorhombic, etc.) are not supported."

    M = [N[d] ÷ 2 for d in 1:dim]
    M2 = [M[d] ÷ 2 for d in 1:dim]
    M2_tuple = Tuple(M2)
    M_tuple = Tuple(M)
    N_tuple = Tuple(N)
    N_vec = collect(N)

    @assert all(M .* 2 .== N_vec) "Grid size must be divisible by 2"
    @assert all(M2 .* 2 .== M) "M must be divisible by 2 for P3c"

    n_spec = length(spec_asu.points)

    # ────────────────────────────────────────────────────────────────
    # Replicate forward plan logic to build A8 table + orbit structure
    # ────────────────────────────────────────────────────────────────

    # Step 1: A8 representative ops (one per parity class)
    subgrid_reps = Vector{Union{Nothing, SymOp}}(nothing, 8)
    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], 2) for d in 1:dim]
        idx = 1 + x0[1] + 2*x0[2] + 4*x0[3]
        subgrid_reps[idx] === nothing && (subgrid_reps[idx] = op)
    end
    active_a8 = [idx for idx in 1:8 if subgrid_reps[idx] !== nothing]

    # Step 2: Remaining point group (even-translation ops)
    rem_ops = [op for op in ops_shifted if all(mod.(round.(Int, op.t), 2) .== 0)]

    # Step 3: Collect G0 positions + A8 raw entries
    g0_pos_set = Set{Int}()
    a8_raw = Vector{Tuple{Int, ComplexF64}}(undef, 8 * n_spec)

    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        base = (h_idx - 1) * 8
        slot = 0
        for a8_idx in active_a8
            g = subgrid_reps[a8_idx]::SymOp
            a8_phase = sum(h_vec[d] * g.t[d] / N[d] for d in 1:dim)
            a8_tw = cispi(-2 * a8_phase)
            rot_h = [mod(sum(Int(g.R[d2,d1]) * h_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]
            lin = 1 + rot_h[1] + M[1]*rot_h[2] + M[1]*M[2]*rot_h[3]
            push!(g0_pos_set, lin)
            slot += 1
            a8_raw[base + slot] = (lin, a8_tw)
        end
        for s in (slot+1):8
            a8_raw[base + s] = (1, complex(0.0))
        end
    end

    # Step 4: Build orbits under G_rem
    g0_to_rep = Dict{Int, Int}()
    g0_to_phase = Dict{Int, ComplexF64}()

    for start_lin in g0_pos_set
        haskey(g0_to_rep, start_lin) && continue
        lin0 = start_lin - 1
        sq = [mod(lin0, M[1]), mod(div(lin0, M[1]), M[2]), div(lin0, M[1]*M[2])]

        orbit = Dict{Int, Tuple{NTuple{3,Int}, ComplexF64}}()
        orbit[start_lin] = (Tuple(sq), complex(1.0))
        worklist = [(sq, complex(1.0))]

        while !isempty(worklist)
            q_vec, q_phase = pop!(worklist)
            for op in rem_ops
                rq = [mod(sum(Int(op.R[d2,d1]) * q_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]
                rlin = 1 + rq[1] + M[1]*rq[2] + M[1]*M[2]*rq[3]
                if !haskey(orbit, rlin)
                    t_half = round.(Int, op.t) .÷ 2
                    sym_phase = cispi(-2 * sum(q_vec[d] * t_half[d] / M[d] for d in 1:dim))
                    orbit[rlin] = (Tuple(rq), sym_phase * q_phase)
                    push!(worklist, (rq, sym_phase * q_phase))
                end
            end
        end

        rep_lin = minimum(keys(orbit))
        rep_phase = orbit[rep_lin][2]
        for (member_lin, (_, member_phase)) in orbit
            g0_to_rep[member_lin] = rep_lin
            g0_to_phase[member_lin] = member_phase / rep_phase
        end
    end

    reps = sort(collect(Set(values(g0_to_rep))))
    rep_to_compact = Dict(r => i for (i, r) in enumerate(reps))
    n_reps = length(reps)

    # ────────────────────────────────────────────────────────────────
    # Build A8 merged table (same as forward) + block-diagonal inverse
    # ────────────────────────────────────────────────────────────────

    # Build A matrix entries per (h, r)
    # A[h, r] = Σ_k a8_tw × sym_phase  where k maps h to g0_pos → rep r
    A_entries = Dict{Tuple{Int,Int}, ComplexF64}()  # (h, compact_r) → weight
    for h in 1:n_spec
        base = (h-1) * 8
        for k in 1:8
            g0_lin, a8_tw = a8_raw[base + k]
            abs(a8_tw) < 1e-14 && continue
            rep_lin = g0_to_rep[g0_lin]
            sym_phase = g0_to_phase[g0_lin]
            compact_r = rep_to_compact[rep_lin]
            key = (h, compact_r)
            A_entries[key] = get(A_entries, key, zero(ComplexF64)) + a8_tw * sym_phase
        end
    end

    # Find blocks via union-find on the coupling graph
    # Two spectral points couple if they share an orbit rep
    rep_to_specs = Dict{Int, Vector{Int}}()
    for ((h, r), _) in A_entries
        if !haskey(rep_to_specs, r); rep_to_specs[r] = Int[]; end
        push!(rep_to_specs[r], h)
    end

    parent = collect(1:n_spec)
    find(x) = (while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end; x)
    function unite(a, b)
        a, b = find(a), find(b)
        a != b && (parent[a] = b)
    end
    for (_, specs) in rep_to_specs
        for i in 2:length(specs)
            unite(specs[1], specs[i])
        end
    end

    # Group into blocks
    block_map = Dict{Int, Vector{Int}}()
    for h in 1:n_spec
        root = find(h)
        if !haskey(block_map, root); block_map[root] = Int[]; end
        push!(block_map[root], h)
    end

    # Build block inverses then flatten into per-rep gather table
    a8_blocks = A8InvBlock[]
    for (_, h_idxs) in block_map
        sort!(h_idxs)
        rep_set = Set{Int}()
        for h in h_idxs
            for ((hh, r), _) in A_entries
                hh == h && push!(rep_set, r)
            end
        end
        rep_list = sort(collect(rep_set))

        n_h = length(h_idxs)
        n_r = length(rep_list)
        @assert n_h == n_r "Block size mismatch: $n_h specs vs $n_r reps"

        rep_to_local = Dict(r => i for (i, r) in enumerate(rep_list))
        A_sub = zeros(ComplexF64, n_h, n_r)
        for (local_h, h) in enumerate(h_idxs)
            for ((hh, r), w) in A_entries
                hh == h && (A_sub[local_h, rep_to_local[r]] += w)
            end
        end

        push!(a8_blocks, A8InvBlock(h_idxs, rep_list, inv(A_sub)))
    end

    # Flatten block-diagonal inverse into per-rep gather table
    # g0_reps[r] = Σ_j inv_A[r_local, j] × F_spec[h_idxs[j]]
    # Padded to 8 entries per rep (matching forward A8 layout)
    inv_a8_table = Vector{InvA8GatherEntry}(undef, 8 * n_reps)
    inv_a8_counts = zeros(Int8, n_reps)
    fill!(inv_a8_table, _ZERO_INV_A8)

    for block in a8_blocks
        inv_A = block.inv_matrix
        h_idxs = block.spec_idxs
        rep_list = block.rep_idxs
        n = length(h_idxs)

        for (local_r, compact_r) in enumerate(rep_list)
            base = (compact_r - 1) * 8
            inv_a8_counts[compact_r] = Int8(n)
            for j in 1:n
                inv_a8_table[base + j] = InvA8GatherEntry(
                    inv_A[local_r, j], Int32(h_idxs[j]))
            end
        end
    end

    # ────────────────────────────────────────────────────────────────
    # Build G_rem expansion table (orbit reps → full M³)
    # ────────────────────────────────────────────────────────────────

    expansion = G0ExpansionEntry[]
    for (member_lin, rep_lin) in g0_to_rep
        compact = rep_to_compact[rep_lin]
        phase = g0_to_phase[member_lin]
        push!(expansion, G0ExpansionEntry(Int32(compact), Int32(member_lin), phase))
    end

    # ────────────────────────────────────────────────────────────────
    # Build per-M2 CSR table (fused expansion + butterfly)
    # For each M2 position, maps expansion entries to their octant.
    # ────────────────────────────────────────────────────────────────

    M2_vol = prod(M2)
    m2_buckets = [OctExpEntry[] for _ in 1:M2_vol]

    for e in expansion
        lin0 = e.target_lin - 1
        x = mod(lin0, M[1]) + 1
        y = mod(div(lin0, M[1]), M[2]) + 1
        z = div(lin0, M[1]*M[2]) + 1

        xi = x > M2[1] ? 1 : 0
        yi = y > M2[2] ? 1 : 0
        zi = z > M2[3] ? 1 : 0
        oct = xi + 2*yi + 4*zi

        mi = xi == 0 ? x : x - M2[1]
        mj = yi == 0 ? y : y - M2[2]
        mk = zi == 0 ? z : z - M2[3]
        m2_lin = mi + (mj-1)*M2[1] + (mk-1)*M2[1]*M2[2]

        push!(m2_buckets[m2_lin], OctExpEntry(Int8(oct), e.rep_compact, e.phase))
    end

    # Build CSR arrays
    total_entries = sum(length(b) for b in m2_buckets)
    row_ptr = zeros(Int32, M2_vol + 1)
    oct_entries = Vector{OctExpEntry}(undef, total_entries)
    pos = 1
    for j in 1:M2_vol
        row_ptr[j] = pos
        for oe in m2_buckets[j]
            oct_entries[pos] = oe
            pos += 1
        end
    end
    row_ptr[M2_vol + 1] = pos
    per_m2 = PerM2Table(row_ptr, oct_entries)

    # ────────────────────────────────────────────────────────────────
    # Butterfly twiddles (same formula as forward)
    # ────────────────────────────────────────────────────────────────

    tw_x = [cispi(-2 * (i-1) / M[1]) for i in 1:M2[1]]
    tw_y = [cispi(-2 * (j-1) / M[2]) for j in 1:M2[2]]
    tw_z = [cispi(-2 * (k-1) / M[3]) for k in 1:M2[3]]

    # ────────────────────────────────────────────────────────────────
    # Symmetry fill map (same as before)
    # ────────────────────────────────────────────────────────────────

    filled = falses(N_tuple)
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1)+1; jj = 4*(j-1)+1; kk = 4*(k-1)+1
        filled[ii, jj, kk] = true
        filled[ii, jj, kk+2] = true
        filled[ii+2, jj+2, kk] = true
        filled[ii+2, jj+2, kk+2] = true
    end

    unfilled_map = Tuple{Int,Int}[]
    for ci in CartesianIndices(N_tuple)
        filled[ci] && continue
        x = collect(Tuple(ci)) .- 1
        for op in ops_shifted
            x2 = mod.(round.(Int, op.R) * x .+ round.(Int, op.t), N_vec)
            ci2 = CartesianIndex(Tuple(x2 .+ 1))
            if filled[ci2]
                push!(unfilled_map, (LinearIndices(N_tuple)[ci], LinearIndices(N_tuple)[ci2]))
                break
            end
        end
    end
    sort!(unfilled_map, by = x -> x[2])   # Opt-1: group by source for cache locality

    # ────────────────────────────────────────────────────────────────
    # Allocate buffers
    # ────────────────────────────────────────────────────────────────

    fft_bufs = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    ifft_out = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    ifft_plan = plan_ifft(fft_bufs[1])
    g0_reps_buf = zeros(ComplexF64, n_reps)

    n_blocks = length(a8_blocks)
    max_block = maximum(length(b.spec_idxs) for b in a8_blocks)
    @info "G0 ASU backward plan (step-inverse): n_spec=$n_spec, n_reps=$n_reps, " *
          "n_blocks=$n_blocks, max_block=$max_block, expansion=$(length(expansion)), " *
          "unfilled=$(length(unfilled_map)), N=$N_tuple"

    return G0ASUBackwardPlan(
        inv_a8_table, inv_a8_counts,
        expansion,
        per_m2,
        tw_x, tw_y, tw_z,
        ifft_plan,
        fft_bufs, ifft_out,
        unfilled_map,
        g0_reps_buf,
        N_vec, M, M2, n_spec, n_reps
    )
end

"""
    execute_g0asu_ikrfft!(plan, spec_asu, F_spec, u_out)

Execute factored backward: F_spec → u_out.

Steps:
1. Inverse A8 (block-diagonal solve): F_spec → g0_reps
2. Fused expansion + butterfly (per-M2 CSR loop): g0_reps → 4×FFT outputs
3. IFFT × 4
4. Unpack stride-4
5. Symmetry fill
"""
function execute_g0asu_ikrfft!(plan::G0ASUBackwardPlan,
                               spec_asu::SpectralIndexing,
                               F_spec::AbstractVector{ComplexF64},
                               u_out::AbstractArray{<:Number,3})
    M2 = plan.sub_sub_dims
    n_reps = plan.n_reps

    # === Step 1: Inverse A8 — flattened gather ===
    g0 = plan.g0_reps
    table = plan.inv_a8_table

    @inbounds for r in 1:n_reps
        base = (r - 1) * 8
        val  = table[base+1].weight * F_spec[table[base+1].spec_idx]
        val += table[base+2].weight * F_spec[table[base+2].spec_idx]
        val += table[base+3].weight * F_spec[table[base+3].spec_idx]
        val += table[base+4].weight * F_spec[table[base+4].spec_idx]
        val += table[base+5].weight * F_spec[table[base+5].spec_idx]
        val += table[base+6].weight * F_spec[table[base+6].spec_idx]
        val += table[base+7].weight * F_spec[table[base+7].spec_idx]
        val += table[base+8].weight * F_spec[table[base+8].spec_idx]
        g0[r] = val
    end

    # === Step 2: Fused expansion + butterfly (per-M2 accumulation) ===
    F000, F001, F110, F111 = plan.fft_bufs
    tw_x, tw_y, tw_z = plan.tw_x, plan.tw_y, plan.tw_z
    row_ptr = plan.per_m2.row_ptr
    entries = plan.per_m2.entries

    @inbounds for k in 1:M2[3]
        cz = conj(tw_z[k])
        for j in 1:M2[2]
            cy = conj(tw_y[j])
            for i in 1:M2[1]
                cx = conj(tw_x[i])
                cxcy = cx * cy
                cxcycz = cxcy * cz

                m2_lin = i + (j-1)*M2[1] + (k-1)*M2[1]*M2[2]

                f000 = zero(ComplexF64)
                f001 = zero(ComplexF64)
                f110 = zero(ComplexF64)
                f111 = zero(ComplexF64)

                for p in row_ptr[m2_lin]:(row_ptr[m2_lin+1]-1)
                    e = entries[p]
                    val = e.phase * g0[e.rep_compact]
                    oct1 = e.octant + 1    # 1-based for sign table

                    f000 += val
                    f001 += _SIGN_F001[oct1] * val
                    f110 += _SIGN_F110[oct1] * val
                    f111 += _SIGN_F111[oct1] * val
                end

                F000[i,j,k] = f000 / 8
                F001[i,j,k] = f001 * cz / 8
                F110[i,j,k] = f110 * cxcy / 8
                F111[i,j,k] = f111 * cxcycz / 8
            end
        end
    end

    # === Step 3: IFFT × 4 ===
    p = plan.ifft_plan
    @inbounds for s in 1:4
        mul!(plan.ifft_out[s], p, plan.fft_bufs[s])
    end

    # === Step 4: Unpack stride-4 ===
    buf000, buf001, buf110, buf111 = plan.ifft_out
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        u_out[ii+1, jj+1, kk+1] = real(buf000[i,j,k])
        u_out[ii+1, jj+1, kk+3] = real(buf001[i,j,k])
        u_out[ii+3, jj+3, kk+1] = real(buf110[i,j,k])
        u_out[ii+3, jj+3, kk+3] = real(buf111[i,j,k])
    end

    # === Step 5: Symmetry fill ===
    u_flat = vec(u_out)
    @inbounds for (target, source) in plan.unfilled_map
        u_flat[target] = u_flat[source]
    end

    return u_out
end
