# Step-level profiling: Forward vs Backward (each step timed individually)

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT
using CrystallographicFFT.KRFFT: plan_krfft_g0asu_backward, G0ASUBackwardPlan
using FFTW
using Statistics
using Random
using LinearAlgebra: mul!
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

function profile_steps(sg, name, N_tuple; n_warmup=10, n_trials=100)
    ops = get_ops(sg, 3, N_tuple)
    _, ops_s = find_optimal_shift(ops, N_tuple)
    spec = calc_spectral_asu(ops_s, 3, N_tuple)
    u = make_symmetric(ops_s, N_tuple)

    fwd = plan_krfft_g0asu(spec, ops_s)
    bwd = plan_krfft_g0asu_backward(spec, ops_s)

    F_asu = execute_g0asu_krfft!(fwd, spec, u)
    F_copy = copy(F_asu)
    M2 = fwd.sub_sub_dims; M2_vol = prod(M2)

    # ── Forward step accessors ──
    p_fwd = fwd.sub_plan
    n_reps_f = fwd.n_reps
    n_spec_f = fwd.n_spec

    # ── Warmup ──
    for _ in 1:n_warmup
        execute_g0asu_krfft!(fwd, spec, u)
    end

    # ── Profile Forward Steps ──
    t_pack = zeros(n_trials)
    t_fft = zeros(n_trials)
    t_concat = zeros(n_trials)
    t_p3c = zeros(n_trials)
    t_a8 = zeros(n_trials)

    for trial in 1:n_trials
        buf000,buf001,buf110,buf111 = fwd.buffers[1],fwd.buffers[2],fwd.buffers[3],fwd.buffers[4]
        t_pack[trial] = @elapsed @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
            ii=4*(i-1); jj=4*(j-1); kk=4*(k-1)
            buf000[i,j,k]=complex(u[ii+1,jj+1,kk+1]); buf001[i,j,k]=complex(u[ii+1,jj+1,kk+3])
            buf110[i,j,k]=complex(u[ii+3,jj+3,kk+1]); buf111[i,j,k]=complex(u[ii+3,jj+3,kk+3])
        end
        t_fft[trial] = @elapsed for i in 1:4; mul!(fwd.work_ffts[i], p_fwd, fwd.buffers[i]); end
        fft_flat = fwd.fft_flat
        t_concat[trial] = @elapsed @inbounds for b in 1:4
            copyto!(fft_flat, (b-1)*M2_vol+1, vec(fwd.work_ffts[b]), 1, M2_vol)
        end
        g0 = fwd.g0_values; g0_p3c = fwd.g0_p3c
        t_p3c[trial] = @elapsed @inbounds for i in 1:n_reps_f
            pw = g0_p3c[i]
            g0[i] = pw.tw[1]*fft_flat[pw.fft_idx[1]] + pw.tw[2]*fft_flat[pw.fft_idx[2]] +
                     pw.tw[3]*fft_flat[pw.fft_idx[3]] + pw.tw[4]*fft_flat[pw.fft_idx[4]] +
                     pw.tw[5]*fft_flat[pw.fft_idx[5]] + pw.tw[6]*fft_flat[pw.fft_idx[6]] +
                     pw.tw[7]*fft_flat[pw.fft_idx[7]] + pw.tw[8]*fft_flat[pw.fft_idx[8]]
        end
        out = fwd.output_buffer; a8 = fwd.a8_table
        t_a8[trial] = @elapsed @inbounds for h in 1:n_spec_f
            base = (h-1)*8
            out[h] = a8[base+1].weight*g0[a8[base+1].g0_idx] + a8[base+2].weight*g0[a8[base+2].g0_idx] +
                     a8[base+3].weight*g0[a8[base+3].g0_idx] + a8[base+4].weight*g0[a8[base+4].g0_idx] +
                     a8[base+5].weight*g0[a8[base+5].g0_idx] + a8[base+6].weight*g0[a8[base+6].g0_idx] +
                     a8[base+7].weight*g0[a8[base+7].g0_idx] + a8[base+8].weight*g0[a8[base+8].g0_idx]
        end
    end

    # ── Profile Backward Steps (ASU-only, no unpack/sym) ──
    # Warmup
    for _ in 1:n_warmup
        copy!(F_copy, F_asu)
        # Just run inv A8 + expand + butterfly + IFFT
        fill!(bwd.g0_reps, 0)
        for block in bwd.a8_blocks
            n=length(block.spec_idxs)
            for j in 1:n; fh=F_copy[block.spec_idxs[j]]
                for i in 1:n; bwd.g0_reps[block.rep_idxs[i]] += block.inv_matrix[i,j]*fh; end
            end
        end
    end

    t_inv_a8 = zeros(n_trials)
    t_expand = zeros(n_trials)
    t_inv_butterfly = zeros(n_trials)
    t_ifft = zeros(n_trials)

    M = bwd.subgrid_dims
    M2b = bwd.sub_sub_dims
    ox, oy, oz = M2b[1], M2b[2], M2b[3]

    for trial in 1:n_trials
        copy!(F_copy, F_asu)

        # Step 1: Inverse A8 (block-diagonal solve)
        g0r = bwd.g0_reps
        t_inv_a8[trial] = @elapsed begin
            fill!(g0r, zero(ComplexF64))
            @inbounds for block in bwd.a8_blocks
                h_idxs = block.spec_idxs; r_idxs = block.rep_idxs
                inv_A = block.inv_matrix; n = length(h_idxs)
                for j in 1:n
                    fh = F_copy[h_idxs[j]]
                    for i in 1:n; g0r[r_idxs[i]] += inv_A[i,j] * fh; end
                end
            end
        end

        # Step 2: G_rem expansion
        g0c = bwd.g0_cache
        t_expand[trial] = @elapsed begin
            fill!(g0c, zero(ComplexF64))
            @inbounds for e in bwd.expansion
                g0c[e.target_lin] = e.phase * g0r[e.rep_compact]
            end
        end

        # Step 3: Inverse butterfly
        w = bwd.work_bufs
        tw_x, tw_y, tw_z = bwd.tw_x, bwd.tw_y, bwd.tw_z
        t_inv_butterfly[trial] = @elapsed begin
            # Read octants
            @inbounds for k in 1:M2b[3], j in 1:M2b[2], i in 1:M2b[1]
                w[1][i,j,k]=g0c[i,j,k];         w[2][i,j,k]=g0c[i,j,k+oz]
                w[3][i,j,k]=g0c[i,j+oy,k];      w[7][i,j,k]=g0c[i,j+oy,k+oz]
                w[4][i,j,k]=g0c[i+ox,j,k];      w[6][i,j,k]=g0c[i+ox,j,k+oz]
                w[5][i,j,k]=g0c[i+ox,j+oy,k];   w[8][i,j,k]=g0c[i+ox,j+oy,k+oz]
            end
            # Inv x
            @inbounds for k in 1:M2b[3], j in 1:M2b[2]
                @simd for i in 1:M2b[1]
                    c=conj(tw_x[i])
                    a=w[1][i,j,k];b=w[4][i,j,k]; w[1][i,j,k]=(a+b)/2;w[4][i,j,k]=(a-b)*c/2
                    a=w[3][i,j,k];b=w[5][i,j,k]; w[3][i,j,k]=(a+b)/2;w[5][i,j,k]=(a-b)*c/2
                    a=w[2][i,j,k];b=w[6][i,j,k]; w[2][i,j,k]=(a+b)/2;w[6][i,j,k]=(a-b)*c/2
                    a=w[7][i,j,k];b=w[8][i,j,k]; w[7][i,j,k]=(a+b)/2;w[8][i,j,k]=(a-b)*c/2
                end
            end
            # Inv y
            @inbounds for k in 1:M2b[3], j in 1:M2b[2]
                c=conj(tw_y[j])
                @simd for i in 1:M2b[1]
                    a=w[1][i,j,k];b=w[3][i,j,k]; w[1][i,j,k]=(a+b)/2;w[3][i,j,k]=(a-b)*c/2
                    a=w[4][i,j,k];b=w[5][i,j,k]; w[4][i,j,k]=(a+b)/2;w[5][i,j,k]=(a-b)*c/2
                    a=w[2][i,j,k];b=w[7][i,j,k]; w[2][i,j,k]=(a+b)/2;w[7][i,j,k]=(a-b)*c/2
                    a=w[6][i,j,k];b=w[8][i,j,k]; w[6][i,j,k]=(a+b)/2;w[8][i,j,k]=(a-b)*c/2
                end
            end
            # Inv z
            @inbounds for k in 1:M2b[3], j in 1:M2b[2]
                c=conj(tw_z[k])
                @simd for i in 1:M2b[1]
                    a=w[1][i,j,k];b=w[2][i,j,k]; w[1][i,j,k]=(a+b)/2;w[2][i,j,k]=(a-b)*c/2
                    a=w[3][i,j,k];b=w[7][i,j,k]; w[3][i,j,k]=(a+b)/2;w[7][i,j,k]=(a-b)*c/2
                    a=w[4][i,j,k];b=w[6][i,j,k]; w[4][i,j,k]=(a+b)/2;w[6][i,j,k]=(a-b)*c/2
                    a=w[5][i,j,k];b=w[8][i,j,k]; w[5][i,j,k]=(a+b)/2;w[8][i,j,k]=(a-b)*c/2
                end
            end
            # Collect
            F000,F001,F110,F111 = bwd.fft_bufs
            @inbounds for k in 1:M2b[3], j in 1:M2b[2], i in 1:M2b[1]
                F000[i,j,k]=w[1][i,j,k]; F001[i,j,k]=w[2][i,j,k]
                F110[i,j,k]=w[5][i,j,k]; F111[i,j,k]=w[8][i,j,k]
            end
        end

        # Step 4: IFFT×4
        ip = bwd.ifft_plan
        t_ifft[trial] = @elapsed @inbounds for s in 1:4
            mul!(bwd.ifft_out[s], ip, bwd.fft_bufs[s])
        end
    end

    # Print
    μ(t) = round(median(t)*1e6, digits=1)
    pct(step, total) = round(100*median(step)/median(total), digits=0)

    fwd_all = t_pack .+ t_fft .+ t_concat .+ t_p3c .+ t_a8
    bwd_all = t_inv_a8 .+ t_expand .+ t_inv_butterfly .+ t_ifft

    println("\n$name (SG $sg) — N=$N_tuple")
    println("  n_spec=$(fwd.n_spec)  n_reps=$(fwd.n_reps)  expansion=$(length(bwd.expansion))")
    println("─"^78)
    println("  FORWARD                       time (μs)    % of total")
    println("    1. pack (stride-4)         $(lpad(μ(t_pack), 10))     $(lpad(pct(t_pack, fwd_all), 4))%")
    println("    2. FFT × 4                 $(lpad(μ(t_fft), 10))     $(lpad(pct(t_fft, fwd_all), 4))%")
    println("    3. concat                  $(lpad(μ(t_concat), 10))     $(lpad(pct(t_concat, fwd_all), 4))%")
    println("    4. P3c (orbit reps)        $(lpad(μ(t_p3c), 10))     $(lpad(pct(t_p3c, fwd_all), 4))%")
    println("    5. A8 recon                $(lpad(μ(t_a8), 10))     $(lpad(pct(t_a8, fwd_all), 4))%")
    println("       TOTAL forward           $(lpad(μ(fwd_all), 10))")
    println()
    println("  BACKWARD (ASU-only)           time (μs)    % of total")
    println("    1. inv A8 (block-solve)    $(lpad(μ(t_inv_a8), 10))     $(lpad(pct(t_inv_a8, bwd_all), 4))%")
    println("    2. G_rem expansion         $(lpad(μ(t_expand), 10))     $(lpad(pct(t_expand, bwd_all), 4))%")
    println("    3. inv butterfly           $(lpad(μ(t_inv_butterfly), 10))     $(lpad(pct(t_inv_butterfly, bwd_all), 4))%")
    println("    4. IFFT × 4                $(lpad(μ(t_ifft), 10))     $(lpad(pct(t_ifft, bwd_all), 4))%")
    println("       TOTAL backward          $(lpad(μ(bwd_all), 10))")
    println()
    println("  bwd/fwd = $(round(μ(bwd_all) / μ(fwd_all), digits=2))×")
end

println("="^78)
println("G0 ASU KRFFT — Step-Level Profiling")
println("="^78)

for (sg, name) in [(225, "Fm-3m")]
    for N in [(16,16,16), (32,32,32), (64,64,64)]
        profile_steps(sg, name, N)
    end
end
