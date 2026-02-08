"""
    RecursiveKRFFT Building Blocks

P3c (cubic 3-fold axis) recursive decomposition.

# Key Design Principle (FFTW-inspired)

Never compute intermediate results larger than the spectral ASU.
For each spectral point, directly gather contributions from the 4 small
sub-sub-grid FFTs using precomputed (buffer_idx, linear_idx, weight) entries.

The fused A8+P3c reconstruction table compresses the two-level decomposition 
into a single flat lookup: each spectral point has n_entries contributions,
where each entry directly references one of the 4 P3c FFT buffers.

# Memory Layout Optimization

Entries are grouped by FFT buffer index for cache-friendly access:
- First, all entries from F_000 (buffer 1)
- Then all entries from F_001 (buffer 2)  
- Then all entries from F_110 (buffer 3)
- Then all entries from F_111 (buffer 4)

This ensures sequential reads from each small FFT buffer.
"""

"""
    P3cReconEntry

One contribution to a spectral ASU point.
"""
struct P3cReconEntry
    linear_idx::Int32      # linear index into FFT buffer
    weight::ComplexF64     # combined A8 + P3c twiddle factor
end

"""
    RecursiveKRFFTPlan

Plan for recursive KRFFT with fused A8 + P3c decomposition.

Uses 4 FFTs of (N/4)³ instead of 1 FFT of (N/2)³ or 1 FFT of N³.
Reconstruction directly targets spectral ASU — no intermediate M³ array.
"""
struct RecursiveKRFFTPlan
    sub_plan::Any                         # FFTW plan for (M/2)³
    buffers::Vector{Array{ComplexF64,3}}  # 4 input: F000, F001, F110, F111
    work_ffts::Vector{Array{ComplexF64,3}} # 4 FFT output (linearized access)
    output_buffer::Vector{ComplexF64}     # spectral ASU output

    # Fused reconstruction: entries grouped by buffer
    # For each spectral point h_idx, entries span:
    #   buffer 1: offsets[1,h_idx] .. offsets[1,h_idx]+counts[1,h_idx]-1
    #   buffer 2: offsets[2,h_idx] .. offsets[2,h_idx]+counts[2,h_idx]-1
    #   etc.
    entries::Vector{P3cReconEntry}        # flat array of all entries
    buf_offsets::Matrix{Int32}            # (4, n_spec): start offset per buffer per point
    buf_counts::Matrix{Int32}            # (4, n_spec): entry count per buffer per point

    grid_N::Vector{Int}
    subgrid_dims::Vector{Int}   # M = N/2
    sub_sub_dims::Vector{Int}   # M2 = M/2 = N/4
    n_a8_ops::Int
end

"""
    plan_krfft_recursive(spec_asu, ops_shifted)

Create a recursive KRFFT plan with fused A8 + P3c decomposition (3D cubic groups only).

The reconstruction table fuses A8 twiddles and P3c index permutations into a single
flat array of (index, weight) entries per spectral point, grouped by FFT buffer for
cache-friendly access. No intermediate arrays larger than the spectral ASU are computed.
"""
function plan_krfft_recursive(spec_asu::SpectralIndexing, ops_shifted::Vector{SymOp})
    N = spec_asu.N
    dim = length(N)
    @assert dim == 3 "Recursive KRFFT currently supports 3D only"

    M = [N[d] ÷ 2 for d in 1:dim]
    M2 = [M[d] ÷ 2 for d in 1:dim]

    @assert all(M .* 2 .== collect(N)) "Grid size must be divisible by 2"
    @assert all(M2 .* 2 .== M) "M must be divisible by 2 for P3c"

    n_spec = length(spec_asu.points)
    n_a8 = 8

    # Select representative operations for 8 A8 subgrids
    subgrid_reps = Vector{SymOp}(undef, n_a8)
    for op in ops_shifted
        t = round.(Int, op.t)
        x0 = [mod(t[d], 2) for d in 1:dim]
        idx = 1 + x0[1] + 2*x0[2] + 4*x0[3]
        subgrid_reps[idx] = op
    end

    # ======================== Build fused reconstruction table ========================
    # Phase 1: Collect all entries per spectral point, grouped by buffer
    # For each h: 8 A8 ops × 8 P3c sub-sub-grids = 64 entries total
    # Each entry maps to one of 4 buffers.

    # Temporary storage: entries_by_buf[buf][h_idx] = [(lin_idx, weight), ...]
    entries_by_buf = [Dict{Int, Vector{Tuple{Int32, ComplexF64}}}() for _ in 1:4]

    for (h_idx, _) in enumerate(spec_asu.points)
        h_vec = get_k_vector(spec_asu, h_idx)
        for buf in 1:4
            entries_by_buf[buf][h_idx] = Tuple{Int32, ComplexF64}[]
        end

        for g in subgrid_reps
            # A8 twiddle
            a8_phase = sum(h_vec[d] * g.t[d] / N[d] for d in 1:dim)
            a8_tw = cispi(-2 * a8_phase)

            # R_g^T h mod M
            rot_h = [mod(sum(Int(g.R[d2, d1]) * h_vec[d2] for d2 in 1:dim), M[d1]) for d1 in 1:dim]

            # P3c contributions from this rotated h
            for l in 0:1, m in 0:1, n in 0:1
                p3c_phase = (rot_h[1]*n/M[1] + rot_h[2]*m/M[2] + rot_h[3]*l/M[3])
                p3c_tw = cispi(-2 * p3c_phase)
                combined_tw = a8_tw * p3c_tw

                # Sub-sub frequency
                ht = [mod(rot_h[d], M2[d]) for d in 1:dim]

                # Map (n,m,l) to buffer + permuted indices
                if n == 0 && m == 0 && l == 0
                    buf = 1; ix, iy, iz = ht[1], ht[2], ht[3]
                elseif n == 1 && m == 0 && l == 0
                    buf = 2; ix, iy, iz = ht[2], ht[3], ht[1]
                elseif n == 0 && m == 1 && l == 0
                    buf = 2; ix, iy, iz = ht[3], ht[1], ht[2]
                elseif n == 0 && m == 0 && l == 1
                    buf = 2; ix, iy, iz = ht[1], ht[2], ht[3]
                elseif n == 1 && m == 1 && l == 0
                    buf = 3; ix, iy, iz = ht[1], ht[2], ht[3]
                elseif n == 1 && m == 0 && l == 1
                    buf = 3; ix, iy, iz = ht[3], ht[1], ht[2]
                elseif n == 0 && m == 1 && l == 1
                    buf = 3; ix, iy, iz = ht[2], ht[3], ht[1]
                else
                    buf = 4; ix, iy, iz = ht[1], ht[2], ht[3]
                end

                lin = Int32(1 + ix + M2[1] * iy + M2[1] * M2[2] * iz)
                push!(entries_by_buf[buf][h_idx], (lin, combined_tw))
            end
        end
    end

    # Phase 2: Merge entries with same (buffer, linear_idx) per spectral point
    for buf in 1:4, h_idx in 1:n_spec
        raw = entries_by_buf[buf][h_idx]
        merged = Dict{Int32, ComplexF64}()
        for (lin, w) in raw
            merged[lin] = get(merged, lin, zero(ComplexF64)) + w
        end
        entries_by_buf[buf][h_idx] = [(k, v) for (k, v) in merged if abs(v) > 1e-15]
    end

    # Phase 3: Pack into flat array with offsets
    total_entries = sum(length(entries_by_buf[buf][h_idx]) for buf in 1:4 for h_idx in 1:n_spec)
    flat_entries = Vector{P3cReconEntry}(undef, total_entries)
    offsets = zeros(Int32, 4, n_spec)
    counts = zeros(Int32, 4, n_spec)

    pos = 1
    for buf in 1:4
        for h_idx in 1:n_spec
            es = entries_by_buf[buf][h_idx]
            offsets[buf, h_idx] = Int32(pos)
            counts[buf, h_idx] = Int32(length(es))
            for (lin, w) in es
                flat_entries[pos] = P3cReconEntry(lin, w)
                pos += 1
            end
        end
    end

    avg_entries = total_entries / n_spec
    @info "P3c plan: n_spec=$n_spec, total_entries=$total_entries, avg/point=$(round(avg_entries, digits=1)) (before merge: 64)"

    # Allocate buffers
    M2_tuple = Tuple(M2)
    buffers = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    work_ffts = [zeros(ComplexF64, M2_tuple) for _ in 1:4]
    sub_plan = plan_fft(buffers[1])
    output_buffer = zeros(ComplexF64, n_spec)

    return RecursiveKRFFTPlan(
        sub_plan, buffers, work_ffts, output_buffer,
        flat_entries, offsets, counts,
        collect(N), M, M2, n_a8
    )
end

"""
    pack_p3c!(plan, u)

Extract 4 P3c sub-sub-grids from real-space array u at stride 4.
"""
function pack_p3c!(plan::RecursiveKRFFTPlan, u::AbstractArray{<:Real})
    M2 = plan.sub_sub_dims
    buf000, buf001, buf110, buf111 = plan.buffers[1], plan.buffers[2], plan.buffers[3], plan.buffers[4]

    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        buf000[i,j,k] = complex(u[ii+1, jj+1, kk+1])
        buf001[i,j,k] = complex(u[ii+1, jj+1, kk+3])
        buf110[i,j,k] = complex(u[ii+3, jj+3, kk+1])
        buf111[i,j,k] = complex(u[ii+3, jj+3, kk+3])
    end
end

"""
    fft_p3c!(plan)

Execute 4 out-of-place FFTs on P3c sub-sub-grids.
"""
function fft_p3c!(plan::RecursiveKRFFTPlan)
    p = plan.sub_plan
    for i in 1:4
        mul!(plan.work_ffts[i], p, plan.buffers[i])
    end
end

"""
    fast_reconstruct_direct!(plan)

Reconstruct spectral ASU directly from 4 P3c FFT buffers.

No intermediate arrays. For each spectral point, reads entries grouped by
FFT buffer for cache-friendly access. Entries with the same (buffer, index)
have been merged during planning, reducing work.
"""
function fast_reconstruct_direct!(plan::RecursiveKRFFTPlan)
    out = plan.output_buffer
    entries = plan.entries
    offsets = plan.buf_offsets
    counts = plan.buf_counts
    n_spec = length(out)

    # Linearize FFT buffers
    fft1 = vec(plan.work_ffts[1])
    fft2 = vec(plan.work_ffts[2])
    fft3 = vec(plan.work_ffts[3])
    fft4 = vec(plan.work_ffts[4])

    @inbounds for h in 1:n_spec
        val = zero(ComplexF64)

        # Buffer 1 (F_000) entries
        off = offsets[1, h]
        cnt = counts[1, h]
        for i in off:off+cnt-1
            e = entries[i]
            val += e.weight * fft1[e.linear_idx]
        end

        # Buffer 2 (F_001) entries
        off = offsets[2, h]
        cnt = counts[2, h]
        for i in off:off+cnt-1
            e = entries[i]
            val += e.weight * fft2[e.linear_idx]
        end

        # Buffer 3 (F_110) entries
        off = offsets[3, h]
        cnt = counts[3, h]
        for i in off:off+cnt-1
            e = entries[i]
            val += e.weight * fft3[e.linear_idx]
        end

        # Buffer 4 (F_111) entries
        off = offsets[4, h]
        cnt = counts[4, h]
        for i in off:off+cnt-1
            e = entries[i]
            val += e.weight * fft4[e.linear_idx]
        end

        out[h] = val
    end

    return out
end

"""
    execute_recursive_krfft!(plan, spec_asu, u)

Full recursive KRFFT pipeline:
1. Pack 4 sub-sub-grids at stride 4
2. FFT × 4 on (N/4)³ grids
3. Directly reconstruct spectral ASU from FFT buffers (no intermediate M³ array)
"""
function execute_recursive_krfft!(plan::RecursiveKRFFTPlan, spec_asu::SpectralIndexing,
                                   u::AbstractArray{<:Real})
    pack_p3c!(plan, u)
    fft_p3c!(plan)
    fast_reconstruct_direct!(plan)
    return plan.output_buffer
end

"""
    reconstruct_g0_at(plan, hx, hy, hz) -> ComplexF64

Reconstruct a single G_0(hx,hy,hz) value from P3c FFTs (debugging only).
"""
@inline function reconstruct_g0_at(plan::RecursiveKRFFTPlan, hx::Int, hy::Int, hz::Int)
    M = plan.subgrid_dims
    M2 = plan.sub_sub_dims

    hxt = mod(hx, M2[1]); hyt = mod(hy, M2[2]); hzt = mod(hz, M2[3])
    ix = hxt + 1; iy = hyt + 1; iz = hzt + 1

    F000, F001, F110, F111 = plan.work_ffts[1], plan.work_ffts[2], plan.work_ffts[3], plan.work_ffts[4]

    val = F000[ix, iy, iz]
    val += cispi(-2*hx/M[1]) * F001[iy, iz, ix]
    val += cispi(-2*hy/M[2]) * F001[iz, ix, iy]
    val += cispi(-2*hz/M[3]) * F001[ix, iy, iz]
    val += cispi(-2*(hx/M[1] + hy/M[2])) * F110[ix, iy, iz]
    val += cispi(-2*(hx/M[1] + hz/M[3])) * F110[iz, ix, iy]
    val += cispi(-2*(hy/M[2] + hz/M[3])) * F110[iy, iz, ix]
    val += cispi(-2*(hx/M[1] + hy/M[2] + hz/M[3])) * F111[ix, iy, iz]

    return val
end
