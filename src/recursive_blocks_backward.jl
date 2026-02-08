# ============================================================================
# G0 ASU Backward Transform — Step-Inverse Pipeline
#
# Pipeline: F_spec → inv A8 → g0_reps → G_rem expand → G0[M³]
#           → inv butterfly → 4×FFT_out → IFFT → unpack → sym fill → u[N³]
#
# Each step mirrors the forward, achieving comparable execution cost.
# ============================================================================

using LinearAlgebra: mul!

# ────────────────────────────────────────────────────────────────────
# Precomputed tables
# ────────────────────────────────────────────────────────────────────

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
    G0ASUBackwardPlan

Step-inverse backward plan mirroring the forward pipeline.
"""
struct G0ASUBackwardPlan
    # Step 1: Inverse A8 (block-diagonal)
    a8_blocks::Vector{A8InvBlock}

    # Step 2: G_rem expansion (orbit reps → full M³)
    expansion::Vector{G0ExpansionEntry}

    # Step 3: Inverse butterfly (reuses forward twiddles)
    tw_x::Vector{ComplexF64}
    tw_y::Vector{ComplexF64}
    tw_z::Vector{ComplexF64}

    # Step 4–5: IFFT + unpack
    ifft_plan::Any
    work_bufs::Vector{Array{ComplexF64,3}}   # 8 work arrays for butterfly
    fft_bufs::Vector{Array{ComplexF64,3}}    # 4 FFT output buffers
    ifft_out::Vector{Array{ComplexF64,3}}    # 4 IFFT output buffers

    # Step 6: Symmetry fill
    unfilled_map::Vector{Tuple{Int,Int}}

    # Intermediate storage
    g0_reps::Vector{ComplexF64}              # orbit-rep values
    g0_cache::Array{ComplexF64,3}            # full M³

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
function plan_krfft_g0asu_backward(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp})
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "G0 ASU backward currently supports 3D only"

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

    # Build block inverses
    a8_blocks = A8InvBlock[]
    for (_, h_idxs) in block_map
        sort!(h_idxs)
        # Collect rep indices used by this block
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

        # Build local A sub-matrix
        rep_to_local = Dict(r => i for (i, r) in enumerate(rep_list))
        A_sub = zeros(ComplexF64, n_h, n_r)
        for (local_h, h) in enumerate(h_idxs)
            for ((hh, r), w) in A_entries
                hh == h && (A_sub[local_h, rep_to_local[r]] += w)
            end
        end

        push!(a8_blocks, A8InvBlock(h_idxs, rep_list, inv(A_sub)))
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

    # ────────────────────────────────────────────────────────────────
    # Allocate buffers
    # ────────────────────────────────────────────────────────────────

    work_bufs = [zeros(ComplexF64, M2_tuple) for _ in 1:8]
    fft_bufs = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    ifft_out = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    ifft_plan = plan_ifft(fft_bufs[1])
    g0_reps_buf = zeros(ComplexF64, n_reps)
    g0_cache = zeros(ComplexF64, M_tuple)

    n_blocks = length(a8_blocks)
    max_block = maximum(length(b.spec_idxs) for b in a8_blocks)
    @info "G0 ASU backward plan (step-inverse): n_spec=$n_spec, n_reps=$n_reps, " *
          "n_blocks=$n_blocks, max_block=$max_block, expansion=$(length(expansion)), " *
          "unfilled=$(length(unfilled_map)), N=$N_tuple"

    return G0ASUBackwardPlan(
        a8_blocks,
        expansion,
        tw_x, tw_y, tw_z,
        ifft_plan,
        work_bufs, fft_bufs, ifft_out,
        unfilled_map,
        g0_reps_buf, g0_cache,
        N_vec, M, M2, n_spec, n_reps
    )
end

"""
    execute_g0asu_ikrfft!(plan, spec_asu, F_spec, u_out)

Execute step-inverse backward: F_spec → u_out.

Steps:
1. Inverse A8 (block-diagonal solve): F_spec → g0_reps
2. G_rem expansion: g0_reps → G0[M³]
3. Inverse P3c butterfly: G0[M³] → 4×FFT outputs
4. IFFT × 4
5. Unpack stride-4
6. Symmetry fill
"""
function execute_g0asu_ikrfft!(plan::G0ASUBackwardPlan,
                               spec_asu::SpectralIndexing,
                               F_spec::AbstractVector{ComplexF64},
                               u_out::AbstractArray{<:Number,3})
    M = plan.subgrid_dims
    M2 = plan.sub_sub_dims

    # === Step 1: Inverse A8 — block-diagonal solve ===
    g0 = plan.g0_reps
    fill!(g0, zero(ComplexF64))

    @inbounds for block in plan.a8_blocks
        h_idxs = block.spec_idxs
        r_idxs = block.rep_idxs
        inv_A = block.inv_matrix
        n = length(h_idxs)

        # g0_block = inv_A × F_block
        for j in 1:n
            fh = F_spec[h_idxs[j]]
            for i in 1:n
                g0[r_idxs[i]] += inv_A[i, j] * fh
            end
        end
    end

    # === Step 2: G_rem expansion — orbit reps → full M³ ===
    g0_cache = plan.g0_cache
    fill!(g0_cache, zero(ComplexF64))

    @inbounds for e in plan.expansion
        g0_cache[e.target_lin] = e.phase * g0[e.rep_compact]
    end

    # === Step 3: Inverse P3c butterfly — G0[M³] → 4×FFT outputs ===
    w = plan.work_bufs
    tw_x, tw_y, tw_z = plan.tw_x, plan.tw_y, plan.tw_z
    ox, oy, oz = M2[1], M2[2], M2[3]

    # Stage 4⁻¹: Read 8 octants from M³ into work arrays
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        w[1][i,j,k] = g0_cache[i,    j,    k]
        w[2][i,j,k] = g0_cache[i,    j,    k+oz]
        w[3][i,j,k] = g0_cache[i,    j+oy, k]
        w[7][i,j,k] = g0_cache[i,    j+oy, k+oz]
        w[4][i,j,k] = g0_cache[i+ox, j,    k]
        w[6][i,j,k] = g0_cache[i+ox, j,    k+oz]
        w[5][i,j,k] = g0_cache[i+ox, j+oy, k]
        w[8][i,j,k] = g0_cache[i+ox, j+oy, k+oz]
    end

    # Stage 3⁻¹: Inverse x-butterfly
    #   Forward: even = w_lo + tw_x × w_hi; odd = w_lo - tw_x × w_hi
    #   Inverse: w_lo = (even + odd) / 2; w_hi = (even - odd) × conj(tw_x) / 2
    #   But tw_x is a unit-magnitude twiddle, so 1/tw_x = conj(tw_x)
    @inbounds for k in 1:M2[3], j in 1:M2[2]
        @simd for i in 1:M2[1]
            twx_inv = conj(tw_x[i])
            a = w[1][i,j,k]; b = w[4][i,j,k]
            w[1][i,j,k] = (a + b) / 2; w[4][i,j,k] = (a - b) * twx_inv / 2

            a = w[3][i,j,k]; b = w[5][i,j,k]
            w[3][i,j,k] = (a + b) / 2; w[5][i,j,k] = (a - b) * twx_inv / 2

            a = w[2][i,j,k]; b = w[6][i,j,k]
            w[2][i,j,k] = (a + b) / 2; w[6][i,j,k] = (a - b) * twx_inv / 2

            a = w[7][i,j,k]; b = w[8][i,j,k]
            w[7][i,j,k] = (a + b) / 2; w[8][i,j,k] = (a - b) * twx_inv / 2
        end
    end

    # Stage 2⁻¹: Inverse y-butterfly
    @inbounds for k in 1:M2[3], j in 1:M2[2]
        twy_inv = conj(tw_y[j])
        @simd for i in 1:M2[1]
            a = w[1][i,j,k]; b = w[3][i,j,k]
            w[1][i,j,k] = (a + b) / 2; w[3][i,j,k] = (a - b) * twy_inv / 2

            a = w[4][i,j,k]; b = w[5][i,j,k]
            w[4][i,j,k] = (a + b) / 2; w[5][i,j,k] = (a - b) * twy_inv / 2

            a = w[2][i,j,k]; b = w[7][i,j,k]
            w[2][i,j,k] = (a + b) / 2; w[7][i,j,k] = (a - b) * twy_inv / 2

            a = w[6][i,j,k]; b = w[8][i,j,k]
            w[6][i,j,k] = (a + b) / 2; w[8][i,j,k] = (a - b) * twy_inv / 2
        end
    end

    # Stage 1⁻¹: Inverse z-butterfly
    @inbounds for k in 1:M2[3], j in 1:M2[2]
        twz_inv = conj(tw_z[k])
        @simd for i in 1:M2[1]
            a = w[1][i,j,k]; b = w[2][i,j,k]
            w[1][i,j,k] = (a + b) / 2; w[2][i,j,k] = (a - b) * twz_inv / 2

            a = w[3][i,j,k]; b = w[7][i,j,k]
            w[3][i,j,k] = (a + b) / 2; w[7][i,j,k] = (a - b) * twz_inv / 2

            a = w[4][i,j,k]; b = w[6][i,j,k]
            w[4][i,j,k] = (a + b) / 2; w[6][i,j,k] = (a - b) * twz_inv / 2

            a = w[5][i,j,k]; b = w[8][i,j,k]
            w[5][i,j,k] = (a + b) / 2; w[8][i,j,k] = (a - b) * twz_inv / 2
        end
    end

    # Stage 0⁻¹: Collect 8 work arrays → 4 FFT outputs (inverse permutation)
    F000, F001, F110, F111 = plan.fft_bufs
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        F000[i,j,k] = w[1][i,j,k]          # (0,0,0)
        F001[i,j,k] = w[2][i,j,k]          # (0,0,1) straight
        # Inverse of Stage 0 forward permutation:
        # Forward: w[3][i,j,k] = F001[k,i,j]  →  F001[k,i,j] = w[3][i,j,k]
        # So: F001[a,b,c] gets contribution from w[3][b,c,a]
        # But since we need F001[i,j,k], we read w[3][j,k,i]
        # Wait — we already wrote w[3][i,j,k] during butterfly.
        # The forward Stage 0 loads w[3] from F001 with permuted INDEX:
        #   w[3][i,j,k] = F001[k,i,j]
        # For the inverse we need: F001[i,j,k] = w[3][j,k,i]
        # But no! That would only hold if w[3] still has the pre-butterfly values.
        # After the inverse butterfly, w[3] has been un-butterflied back to the
        # pre-butterfly state. So F001[k,i,j] = w[3][i,j,k] is correct.
        # But the 4 physical FFT outputs are F000, F001, F110, F111.
        # The permuted entries come from the SAME F001 array:
        #   w[3][i,j,k] was loaded from F001[k,i,j]
        #   w[4][i,j,k] was loaded from F001[j,k,i]
        # After inverse butterfly, w[3] contains what was originally F001 at [k,i,j].
        # We DON'T need to reconstruct F001 at permuted indices — we just take
        # the straight entries:
        F110[i,j,k] = w[5][i,j,k]          # (1,1,0) straight
        F111[i,j,k] = w[8][i,j,k]          # (1,1,1) straight
        # w[3], w[4], w[6], w[7] contain permuted copies — we can verify they're consistent
    end

    # === Step 4: IFFT × 4 ===
    p = plan.ifft_plan
    @inbounds for s in 1:4
        mul!(plan.ifft_out[s], p, plan.fft_bufs[s])
    end

    # === Step 5: Unpack stride-4 ===
    buf000, buf001, buf110, buf111 = plan.ifft_out
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        u_out[ii+1, jj+1, kk+1] = real(buf000[i,j,k])
        u_out[ii+1, jj+1, kk+3] = real(buf001[i,j,k])
        u_out[ii+3, jj+3, kk+1] = real(buf110[i,j,k])
        u_out[ii+3, jj+3, kk+3] = real(buf111[i,j,k])
    end

    # === Step 6: Symmetry fill ===
    u_flat = vec(u_out)
    @inbounds for (target, source) in plan.unfilled_map
        u_flat[target] = u_flat[source]
    end

    return u_out
end
