# Fine-grained per-step timing: forward vs backward, all sub-steps
# Identifies engineering-level optimization opportunities

using FFTW, Random
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT
using CrystallographicFFT.KRFFT: plan_krfft_g0asu_backward, execute_g0asu_ikrfft!,
    G0ASUPlan, G0ASUBackwardPlan
using LinearAlgebra: mul!
using Statistics
using Printf

function make_symmetric(ops, N)
    Random.seed!(42)
    u = rand(N...); s = zeros(N...)
    for op in ops, idx in CartesianIndices(u)
        x = collect(Tuple(idx)) .- 1
        x2 = mod.(round.(Int, op.R) * x .+ round.(Int, op.t), collect(N)) .+ 1
        s[idx] += u[x2...]
    end
    s ./= length(ops)
end

# ================================================================
# Forward sub-step functions (matching execute_g0asu_krfft! exactly)
# ================================================================

function fwd_pack!(plan, u)
    M2 = plan.sub_sub_dims
    buf000, buf001 = plan.buffers[1], plan.buffers[2]
    buf110, buf111 = plan.buffers[3], plan.buffers[4]
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        buf000[i,j,k] = complex(u[ii+1, jj+1, kk+1])
        buf001[i,j,k] = complex(u[ii+1, jj+1, kk+3])
        buf110[i,j,k] = complex(u[ii+3, jj+3, kk+1])
        buf111[i,j,k] = complex(u[ii+3, jj+3, kk+3])
    end
end

function fwd_fft!(plan)
    p = plan.sub_plan
    @inbounds for i in 1:4; mul!(plan.work_ffts[i], p, plan.buffers[i]); end
end

function fwd_concat!(plan)
    M2_vol = prod(plan.sub_sub_dims)
    fft = plan.fft_flat
    @inbounds for b in 1:4
        offset = (b-1) * M2_vol
        copyto!(fft, offset+1, vec(plan.work_ffts[b]), 1, M2_vol)
    end
end

function fwd_p3c!(plan)
    g0 = plan.g0_values; g0_p3c = plan.g0_p3c; fft = plan.fft_flat
    @inbounds for i in 1:plan.n_reps
        pw = g0_p3c[i]
        val  = pw.tw[1] * fft[pw.fft_idx[1]]
        val += pw.tw[2] * fft[pw.fft_idx[2]]
        val += pw.tw[3] * fft[pw.fft_idx[3]]
        val += pw.tw[4] * fft[pw.fft_idx[4]]
        val += pw.tw[5] * fft[pw.fft_idx[5]]
        val += pw.tw[6] * fft[pw.fft_idx[6]]
        val += pw.tw[7] * fft[pw.fft_idx[7]]
        val += pw.tw[8] * fft[pw.fft_idx[8]]
        g0[i] = val
    end
end

function fwd_a8!(plan)
    out = plan.output_buffer; a8 = plan.a8_table; g0 = plan.g0_values
    @inbounds for h in 1:plan.n_spec
        base = (h-1) * 8
        val  = a8[base+1].weight * g0[a8[base+1].g0_idx]
        val += a8[base+2].weight * g0[a8[base+2].g0_idx]
        val += a8[base+3].weight * g0[a8[base+3].g0_idx]
        val += a8[base+4].weight * g0[a8[base+4].g0_idx]
        val += a8[base+5].weight * g0[a8[base+5].g0_idx]
        val += a8[base+6].weight * g0[a8[base+6].g0_idx]
        val += a8[base+7].weight * g0[a8[base+7].g0_idx]
        val += a8[base+8].weight * g0[a8[base+8].g0_idx]
        out[h] = val
    end
end

# ================================================================
# Backward sub-step functions (matching execute_g0asu_ikrfft! exactly)
# ================================================================

function bwd_inv_a8!(plan, F_spec)
    g0 = plan.g0_reps; fill!(g0, zero(ComplexF64))
    @inbounds for block in plan.a8_blocks
        h_idxs = block.spec_idxs; r_idxs = block.rep_idxs
        inv_A = block.inv_matrix; n = length(h_idxs)
        for j in 1:n; fh = F_spec[h_idxs[j]]
            for i in 1:n; g0[r_idxs[i]] += inv_A[i,j] * fh; end
        end
    end
end

function bwd_fused_expand_butterfly!(plan)
    M2 = plan.sub_sub_dims; g0 = plan.g0_reps
    F000, F001, F110, F111 = plan.fft_bufs
    tw_x, tw_y, tw_z = plan.tw_x, plan.tw_y, plan.tw_z
    row_ptr = plan.per_m2.row_ptr; entries = plan.per_m2.entries

    @inbounds for k in 1:M2[3]
        cz = conj(tw_z[k])
        for j in 1:M2[2]
            cy = conj(tw_y[j])
            for i in 1:M2[1]
                cx = conj(tw_x[i])
                cxcy = cx * cy; cxcycz = cxcy * cz
                m2_lin = i + (j-1)*M2[1] + (k-1)*M2[1]*M2[2]
                f000 = zero(ComplexF64); f001 = zero(ComplexF64)
                f110 = zero(ComplexF64); f111 = zero(ComplexF64)
                for p in row_ptr[m2_lin]:(row_ptr[m2_lin+1]-1)
                    e = entries[p]
                    val = e.phase * g0[e.rep_compact]
                    oct1 = e.octant + 1
                    f000 += val
                    f001 += CrystallographicFFT.KRFFT._SIGN_F001[oct1] * val
                    f110 += CrystallographicFFT.KRFFT._SIGN_F110[oct1] * val
                    f111 += CrystallographicFFT.KRFFT._SIGN_F111[oct1] * val
                end
                F000[i,j,k] = f000 / 8
                F001[i,j,k] = f001 * cz / 8
                F110[i,j,k] = f110 * cxcy / 8
                F111[i,j,k] = f111 * cxcycz / 8
            end
        end
    end
end

function bwd_ifft!(plan)
    p = plan.ifft_plan
    @inbounds for s in 1:4; mul!(plan.ifft_out[s], p, plan.fft_bufs[s]); end
end

function bwd_unpack!(plan, u_out)
    M2 = plan.sub_sub_dims
    buf000, buf001, buf110, buf111 = plan.ifft_out
    @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii = 4*(i-1); jj = 4*(j-1); kk = 4*(k-1)
        u_out[ii+1,jj+1,kk+1] = real(buf000[i,j,k])
        u_out[ii+1,jj+1,kk+3] = real(buf001[i,j,k])
        u_out[ii+3,jj+3,kk+1] = real(buf110[i,j,k])
        u_out[ii+3,jj+3,kk+3] = real(buf111[i,j,k])
    end
end

function bwd_symfill!(plan, u_out)
    u_flat = vec(u_out)
    @inbounds for (target, source) in plan.unfilled_map
        u_flat[target] = u_flat[source]
    end
end

# ================================================================
# Profiling
# ================================================================

function profile(sg, name, N_tuple; n_trials=500)
    ops = get_ops(sg, 3, N_tuple)
    _, ops_s = find_optimal_shift(ops, N_tuple)
    spec = calc_spectral_asu(ops_s, 3, N_tuple)
    u = make_symmetric(ops_s, N_tuple)
    u_out = zeros(N_tuple...)

    fwd = plan_krfft_g0asu(spec, ops_s)
    bwd = plan_krfft_g0asu_backward(spec, ops_s)
    F_spec = copy(execute_g0asu_krfft!(fwd, spec, u))

    M2 = fwd.sub_sub_dims; M2_vol = prod(M2)
    n_reps = fwd.n_reps; n_spec = fwd.n_spec

    # Warmup
    for _ in 1:20
        fwd_pack!(fwd, u); fwd_fft!(fwd); fwd_concat!(fwd)
        fwd_p3c!(fwd); fwd_a8!(fwd)
        bwd_inv_a8!(bwd, F_spec); bwd_fused_expand_butterfly!(bwd)
        bwd_ifft!(bwd); bwd_unpack!(bwd, u_out); bwd_symfill!(bwd, u_out)
    end

    # Allocations
    a_pack = @allocated fwd_pack!(fwd, u)
    a_fft = @allocated fwd_fft!(fwd)
    a_concat = @allocated fwd_concat!(fwd)
    a_p3c = @allocated fwd_p3c!(fwd)
    a_a8f = @allocated fwd_a8!(fwd)
    a_ia8 = @allocated bwd_inv_a8!(bwd, F_spec)
    a_fused = @allocated bwd_fused_expand_butterfly!(bwd)
    a_ifft = @allocated bwd_ifft!(bwd)
    a_unpack = @allocated bwd_unpack!(bwd, u_out)
    a_sym = @allocated bwd_symfill!(bwd, u_out)

    # Timing
    tf_pack = zeros(n_trials); tf_fft = zeros(n_trials); tf_concat = zeros(n_trials)
    tf_p3c = zeros(n_trials); tf_a8 = zeros(n_trials)
    tb_ia8 = zeros(n_trials); tb_fused = zeros(n_trials); tb_ifft = zeros(n_trials)
    tb_unpack = zeros(n_trials); tb_sym = zeros(n_trials)

    for t in 1:n_trials
        tf_pack[t] = @elapsed fwd_pack!(fwd, u)
        tf_fft[t] = @elapsed fwd_fft!(fwd)
        tf_concat[t] = @elapsed fwd_concat!(fwd)
        tf_p3c[t] = @elapsed fwd_p3c!(fwd)
        tf_a8[t] = @elapsed fwd_a8!(fwd)

        tb_ia8[t] = @elapsed bwd_inv_a8!(bwd, F_spec)
        tb_fused[t] = @elapsed bwd_fused_expand_butterfly!(bwd)
        tb_ifft[t] = @elapsed bwd_ifft!(bwd)
        tb_unpack[t] = @elapsed bwd_unpack!(bwd, u_out)
        tb_sym[t] = @elapsed bwd_symfill!(bwd, u_out)
    end

    μ(t) = round(median(t)*1e6, digits=1)

    tf_total = tf_pack .+ tf_fft .+ tf_concat .+ tf_p3c .+ tf_a8
    tb_total = tb_ia8 .+ tb_fused .+ tb_ifft .+ tb_unpack .+ tb_sym
    ft = μ(tf_total); bt = μ(tb_total)

    n_entries = length(bwd.per_m2.entries)
    avg_per_m2 = round(n_entries / M2_vol, digits=1)

    println("\n$name  N=$(N_tuple)  n_spec=$n_spec  n_reps=$n_reps  M2³=$M2_vol  entries=$n_entries  avg/M2=$avg_per_m2")
    println("─"^90)
    @printf("  FORWARD    %7.1f μs                     BACKWARD   %7.1f μs    ratio: %.2f×\n", ft, bt, bt/ft)
    println("─"^90)
    @printf("  %-18s %7.1f μs %5.1f%% %5d B    %-18s %7.1f μs %5.1f%% %5d B\n",
        "pack (4×M2³ rd)", μ(tf_pack), 100*μ(tf_pack)/ft, a_pack,
        "inv A8 (blk solve)", μ(tb_ia8), 100*μ(tb_ia8)/bt, a_ia8)
    @printf("  %-18s %7.1f μs %5.1f%% %5d B    %-18s %7.1f μs %5.1f%% %5d B\n",
        "FFT ×4", μ(tf_fft), 100*μ(tf_fft)/ft, a_fft,
        "fused exp+bfly", μ(tb_fused), 100*μ(tb_fused)/bt, a_fused)
    @printf("  %-18s %7.1f μs %5.1f%% %5d B    %-18s %7.1f μs %5.1f%% %5d B\n",
        "concat", μ(tf_concat), 100*μ(tf_concat)/ft, a_concat,
        "IFFT ×4", μ(tb_ifft), 100*μ(tb_ifft)/bt, a_ifft)
    @printf("  %-18s %7.1f μs %5.1f%% %5d B    %-18s %7.1f μs %5.1f%% %5d B\n",
        "P3c (8-gather)", μ(tf_p3c), 100*μ(tf_p3c)/ft, a_p3c,
        "unpack (4×M2³ wr)", μ(tb_unpack), 100*μ(tb_unpack)/bt, a_unpack)
    @printf("  %-18s %7.1f μs %5.1f%% %5d B    %-18s %7.1f μs %5.1f%% %5d B\n",
        "A8 (8-gather)", μ(tf_a8), 100*μ(tf_a8)/ft, a_a8f,
        "sym fill", μ(tb_sym), 100*μ(tb_sym)/bt, a_sym)
    println("─"^90)

    # Cross-pipeline comparison for symmetric steps
    fft_ratio = μ(tb_ifft) / μ(tf_fft)
    pack_ratio = μ(tb_unpack) / μ(tf_pack)
    a8_ratio = μ(tb_ia8) / μ(tf_a8)
    gather_fwd = μ(tf_p3c) + μ(tf_a8)
    gather_bwd = μ(tb_fused)

    println("  Step-by-step ratios:")
    @printf("    FFT vs IFFT:           %.2f×  (%.1f vs %.1f μs)\n", fft_ratio, μ(tf_fft), μ(tb_ifft))
    @printf("    pack vs unpack:        %.2f×  (%.1f vs %.1f μs)\n", pack_ratio, μ(tf_pack), μ(tb_unpack))
    @printf("    A8 vs inv A8:          %.2f×  (%.1f vs %.1f μs)\n", a8_ratio, μ(tf_a8), μ(tb_ia8))
    @printf("    P3c+A8 vs fused:       %.2f×  (%.1f vs %.1f μs)  ← main asymmetry\n",
        gather_bwd / gather_fwd, gather_fwd, gather_bwd)

    # Per-entry analysis of the fused loop
    us_per_entry = μ(tb_fused) / n_entries * 1000  # ns per entry
    us_per_m2 = μ(tb_fused) / M2_vol * 1000        # ns per M2 point
    @printf("\n  Per-entry metrics:\n")
    @printf("    fused: %.1f ns/entry, %.1f ns/M2-point\n", us_per_entry, us_per_m2)
    @printf("    P3c:   %.1f ns/rep (8 gathers)\n", μ(tf_p3c) / n_reps * 1000)
    @printf("    A8:    %.1f ns/spec (8 gathers)\n", μ(tf_a8) / n_spec * 1000)

    # Memory footprint
    g0_reps_kb = n_reps * 16 / 1024
    table_kb = n_entries * (1 + 4 + 16) / 1024  # Int8 + Int32 + ComplexF64
    fft_bufs_kb = 4 * M2_vol * 16 / 1024
    @printf("\n  Cache analysis:\n")
    @printf("    g0_reps: %.1f KB  (%s)\n", g0_reps_kb,
        g0_reps_kb < 32 ? "L1" : g0_reps_kb < 256 ? "L2" : "L3")
    @printf("    per-M2 table: %.1f KB  (%s)\n", table_kb,
        table_kb < 32 ? "L1" : table_kb < 256 ? "L2" : "L3")
    @printf("    fft_bufs (4×): %.1f KB  (%s)\n", fft_bufs_kb,
        fft_bufs_kb < 32 ? "L1" : fft_bufs_kb < 256 ? "L2" : "L3")
    @printf("    fwd fft_flat: %.1f KB\n", 4 * M2_vol * 16 / 1024)
end

println("="^90)
println("Fine-grained per-step profiling: forward vs backward")
println("="^90)
for (sg, name) in [(225, "Fm-3m"), (229, "Im-3m"), (221, "Pm-3m"), (200, "Pm-3")]
    for N in [(16,16,16), (32,32,32), (64,64,64)]
        profile(sg, name, N)
    end
end
