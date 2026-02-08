# Factored Per-M2 Accumulation: inv A8 (separate) + expansion+butterfly (fused per-M2)
#
# Key idea: g0_reps (26KB) fits in L1, table = |expansion| entries (115KB) fits in L2
# For each M2 position, gather sparse octant values and apply 4 butterfly formulas

using FFTW, Random
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.KRFFT
using CrystallographicFFT.KRFFT: plan_krfft_g0asu_backward
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

# Per-M2 octant expansion entry
struct OctExpEntry
    octant::Int8       # 0-7 octant index
    rep_compact::Int32
    phase::ComplexF64
end

# Octant-to-slot mapping (from fill_g0_butterfly! Stage 4)
# (xi,yi,zi) → butterfly work slot
const OCT_TO_SLOT = Int8[
    1,  # (0,0,0) → slot 1
    2,  # (0,0,1) → slot 2
    3,  # (0,1,0) → slot 3
    7,  # (0,1,1) → slot 7
    4,  # (1,0,0) → slot 4
    6,  # (1,0,1) → slot 6
    5,  # (1,1,0) → slot 5
    8,  # (1,1,1) → slot 8
]

# Sign factors for each sub-FFT output from each octant (oct = xi + 2*yi + 4*zi)
# oct:  0:(000) 1:(100) 2:(010) 3:(110) 4:(001) 5:(101) 6:(011) 7:(111)
# F000: always +1
# F001: (-1)^zi   → +1,+1,+1,+1,-1,-1,-1,-1
# F110: (-1)^(xi⊕yi) → +1,-1,-1,+1,+1,-1,-1,+1
# F111: (-1)^(xi⊕yi⊕zi) → +1,-1,-1,+1,-1,+1,+1,-1
const SIGN_F001 = Int8[ 1, 1, 1, 1,-1,-1,-1,-1]
const SIGN_F110 = Int8[ 1,-1,-1, 1, 1,-1,-1, 1]
const SIGN_F111 = Int8[ 1,-1,-1, 1,-1, 1, 1,-1]

struct PerM2Table
    row_ptr::Vector{Int32}          # M2_vol + 1
    entries::Vector{OctExpEntry}
end

function build_per_m2_table(expansion, M, M2, M2_vol)
    # Collect entries per M2 position
    buckets = [OctExpEntry[] for _ in 1:M2_vol]

    for e in expansion
        lin0 = e.target_lin - 1
        x = mod(lin0, M[1]) + 1
        y = mod(div(lin0, M[1]), M[2]) + 1
        z = div(lin0, M[1]*M[2]) + 1

        xi = x > M2[1] ? 1 : 0
        yi = y > M2[2] ? 1 : 0
        zi = z > M2[3] ? 1 : 0
        oct = xi + 2*yi + 4*zi   # 0-based octant index

        mi = xi == 0 ? x : x - M2[1]
        mj = yi == 0 ? y : y - M2[2]
        mk = zi == 0 ? z : z - M2[3]
        m2_lin = mi + (mj-1)*M2[1] + (mk-1)*M2[1]*M2[2]

        push!(buckets[m2_lin], OctExpEntry(Int8(oct), e.rep_compact, e.phase))
    end

    # Build CSR
    total = sum(length(b) for b in buckets)
    row_ptr = zeros(Int32, M2_vol + 1)
    entries = Vector{OctExpEntry}(undef, total)
    pos = 1
    for j in 1:M2_vol
        row_ptr[j] = pos
        for e in buckets[j]
            entries[pos] = e
            pos += 1
        end
    end
    row_ptr[M2_vol + 1] = pos
    return PerM2Table(row_ptr, entries)
end

# Execute: per-M2 accumulation with closed-form butterfly
function execute_factored!(F000, F001, F110, F111,
                           g0_reps, table, tw_x, tw_y, tw_z, M2)
    row_ptr = table.row_ptr
    entries = table.entries

    @inbounds for k in 1:M2[3]
        cz = conj(tw_z[k])
        for j in 1:M2[2]
            cy = conj(tw_y[j])
            cxcy_base = cy  # will multiply by cx inside
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
                    val = e.phase * g0_reps[e.rep_compact]
                    oct1 = e.octant + 1   # 1-based for indexing sign tables

                    f000 += val
                    f001 += SIGN_F001[oct1] * val
                    f110 += SIGN_F110[oct1] * val
                    f111 += SIGN_F111[oct1] * val
                end

                F000[i,j,k] = f000 / 8
                F001[i,j,k] = f001 * cz / 8
                F110[i,j,k] = f110 * cxcy / 8
                F111[i,j,k] = f111 * cxcycz / 8
            end
        end
    end
end

# Inv A8 step (same as current)
function step_inv_a8!(g0_reps, a8_blocks, F_spec)
    fill!(g0_reps, zero(ComplexF64))
    @inbounds for block in a8_blocks
        h_idxs = block.spec_idxs; r_idxs = block.rep_idxs
        inv_A = block.inv_matrix; n = length(h_idxs)
        for j in 1:n
            fh = F_spec[h_idxs[j]]
            for i in 1:n; g0_reps[r_idxs[i]] += inv_A[i,j] * fh; end
        end
    end
end

function bench(sg, name, N_tuple; n_warmup=10, n_trials=200)
    ops = get_ops(sg, 3, N_tuple)
    _, ops_s = find_optimal_shift(ops, N_tuple)
    spec = calc_spectral_asu(ops_s, 3, N_tuple)
    u = make_symmetric(ops_s, N_tuple)

    fwd = plan_krfft_g0asu(spec, ops_s)
    bwd = plan_krfft_g0asu_backward(spec, ops_s)
    F_spec = copy(execute_g0asu_krfft!(fwd, spec, u))

    M = bwd.subgrid_dims; M2 = bwd.sub_sub_dims; M2_vol = prod(M2)
    tw_x, tw_y, tw_z = bwd.tw_x, bwd.tw_y, bwd.tw_z
    fft_bufs = bwd.fft_bufs
    ifft_out = bwd.ifft_out; ifft_plan = bwd.ifft_plan

    # Build per-M2 table
    table = build_per_m2_table(bwd.expansion, M, M2, M2_vol)
    n_entries = length(table.entries)

    # Verify correctness first
    step_inv_a8!(bwd.g0_reps, bwd.a8_blocks, F_spec)
    execute_factored!(fft_bufs[1], fft_bufs[2], fft_bufs[3], fft_bufs[4],
                      bwd.g0_reps, table, tw_x, tw_y, tw_z, M2)
    for s in 1:4; mul!(ifft_out[s], ifft_plan, fft_bufs[s]); end
    max_err = 0.0
    for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
        ii=4*(i-1); jj=4*(j-1); kk=4*(k-1)
        max_err = max(max_err, abs(u[ii+1,jj+1,kk+1] - real(ifft_out[1][i,j,k])))
        max_err = max(max_err, abs(u[ii+1,jj+1,kk+3] - real(ifft_out[2][i,j,k])))
        max_err = max(max_err, abs(u[ii+3,jj+3,kk+1] - real(ifft_out[3][i,j,k])))
        max_err = max(max_err, abs(u[ii+3,jj+3,kk+3] - real(ifft_out[4][i,j,k])))
    end

    # Warmup
    for _ in 1:n_warmup
        step_inv_a8!(bwd.g0_reps, bwd.a8_blocks, F_spec)
        execute_factored!(fft_bufs[1], fft_bufs[2], fft_bufs[3], fft_bufs[4],
                          bwd.g0_reps, table, tw_x, tw_y, tw_z, M2)
    end

    # === New: factored per-M2 ===
    t_new_a8 = zeros(n_trials); t_new_bfly = zeros(n_trials); t_new_ifft = zeros(n_trials)
    for trial in 1:n_trials
        t_new_a8[trial] = @elapsed step_inv_a8!(bwd.g0_reps, bwd.a8_blocks, F_spec)
        t_new_bfly[trial] = @elapsed execute_factored!(
            fft_bufs[1], fft_bufs[2], fft_bufs[3], fft_bufs[4],
            bwd.g0_reps, table, tw_x, tw_y, tw_z, M2)
        t_new_ifft[trial] = @elapsed begin
            for s in 1:4; mul!(ifft_out[s], ifft_plan, fft_bufs[s]); end
        end
    end

    # === Current: step-inverse backward ===
    w1,w2,w3,w4,w5,w6,w7,w8 = bwd.work_bufs; ox,oy,oz = M2
    g0_cache = bwd.g0_cache

    t_cur = zeros(n_trials)
    for trial in 1:n_trials
        t_cur[trial] = @elapsed begin
            step_inv_a8!(bwd.g0_reps, bwd.a8_blocks, F_spec)
            fill!(g0_cache, zero(ComplexF64))
            @inbounds for e in bwd.expansion; g0_cache[e.target_lin] = e.phase * bwd.g0_reps[e.rep_compact]; end
            @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
                w1[i,j,k]=g0_cache[i,j,k]; w2[i,j,k]=g0_cache[i,j,k+oz]
                w3[i,j,k]=g0_cache[i,j+oy,k]; w7[i,j,k]=g0_cache[i,j+oy,k+oz]
                w4[i,j,k]=g0_cache[i+ox,j,k]; w6[i,j,k]=g0_cache[i+ox,j,k+oz]
                w5[i,j,k]=g0_cache[i+ox,j+oy,k]; w8[i,j,k]=g0_cache[i+ox,j+oy,k+oz]
            end
            @inbounds for k in 1:M2[3], j in 1:M2[2]
                @simd for i in 1:M2[1]
                    c=conj(tw_x[i])
                    a=w1[i,j,k];b=w4[i,j,k]; w1[i,j,k]=(a+b)/2;w4[i,j,k]=(a-b)*c/2
                    a=w3[i,j,k];b=w5[i,j,k]; w3[i,j,k]=(a+b)/2;w5[i,j,k]=(a-b)*c/2
                    a=w2[i,j,k];b=w6[i,j,k]; w2[i,j,k]=(a+b)/2;w6[i,j,k]=(a-b)*c/2
                    a=w7[i,j,k];b=w8[i,j,k]; w7[i,j,k]=(a+b)/2;w8[i,j,k]=(a-b)*c/2
                end
            end
            @inbounds for k in 1:M2[3], j in 1:M2[2]
                c=conj(tw_y[j]); @simd for i in 1:M2[1]
                    a=w1[i,j,k];b=w3[i,j,k]; w1[i,j,k]=(a+b)/2;w3[i,j,k]=(a-b)*c/2
                    a=w4[i,j,k];b=w5[i,j,k]; w4[i,j,k]=(a+b)/2;w5[i,j,k]=(a-b)*c/2
                    a=w2[i,j,k];b=w7[i,j,k]; w2[i,j,k]=(a+b)/2;w7[i,j,k]=(a-b)*c/2
                    a=w6[i,j,k];b=w8[i,j,k]; w6[i,j,k]=(a+b)/2;w8[i,j,k]=(a-b)*c/2
                end
            end
            @inbounds for k in 1:M2[3], j in 1:M2[2]
                c=conj(tw_z[k]); @simd for i in 1:M2[1]
                    a=w1[i,j,k];b=w2[i,j,k]; w1[i,j,k]=(a+b)/2;w2[i,j,k]=(a-b)*c/2
                    a=w3[i,j,k];b=w7[i,j,k]; w3[i,j,k]=(a+b)/2;w7[i,j,k]=(a-b)*c/2
                    a=w4[i,j,k];b=w6[i,j,k]; w4[i,j,k]=(a+b)/2;w6[i,j,k]=(a-b)*c/2
                    a=w5[i,j,k];b=w8[i,j,k]; w5[i,j,k]=(a+b)/2;w8[i,j,k]=(a-b)*c/2
                end
            end
            @inbounds for k in 1:M2[3], j in 1:M2[2], i in 1:M2[1]
                fft_bufs[1][i,j,k]=w1[i,j,k]; fft_bufs[2][i,j,k]=w2[i,j,k]
                fft_bufs[3][i,j,k]=w5[i,j,k]; fft_bufs[4][i,j,k]=w8[i,j,k]
            end
            for s in 1:4; mul!(ifft_out[s], ifft_plan, fft_bufs[s]); end
        end
    end

    # === Forward ===
    t_fwd = [(@elapsed execute_g0asu_krfft!(fwd, spec, u)) for _ in 1:n_trials]

    μ(t) = round(median(t)*1e6, digits=1)
    f = μ(t_fwd); c = μ(t_cur)
    new_total = μ(t_new_a8 .+ t_new_bfly .+ t_new_ifft)
    @printf("%-6s %-14s  entries=%5d  err=%.1e  fwd=%7.1f  cur=%7.1f  new=%7.1f (a8=%.1f bfly=%.1f ifft=%.1f)  new/fwd=%.2f  speedup=%.2fx\n",
            name, string(N_tuple), n_entries, max_err,
            f, c, new_total, μ(t_new_a8), μ(t_new_bfly), μ(t_new_ifft),
            new_total/f, c/new_total)
end

println("="^150)
println("Factored Per-M2 Accumulation Benchmark")
println("="^150)
for (sg, name) in [(225, "Fm-3m"), (229, "Im-3m"), (221, "Pm-3m"), (200, "Pm-3")]
    for N in [(16,16,16), (32,32,32), (64,64,64)]
        bench(sg, name, N)
    end
    println()
end
