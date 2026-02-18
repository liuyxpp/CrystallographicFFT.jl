# ============================================================================
# Benchmark: rcfft!/ircfft! vs cfft!/icfft!
#
# Fair comparison — benchmarks internal execution functions directly:
#   Forward:  fft_reconstruct!(complex buf) vs rfft_reconstruct!(real buf)
#   Backward: execute_backward!(→complex)  vs execute_real_backward!(→real)
#
# No BenchmarkTools dependency — uses manual warmup + timing.
# ============================================================================

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.CFFTApi
using FFTW
using LinearAlgebra: mul!
using Printf
using Statistics

include("test_helpers.jl")

const N_WARMUP = 5
const N_TRIALS = 30

function bench(f, args...; n_warmup=N_WARMUP, n_trials=N_TRIALS)
    for _ in 1:n_warmup
        f(args...)
    end
    times = Vector{Float64}(undef, n_trials)
    for i in 1:n_trials
        t0 = time_ns()
        f(args...)
        times[i] = (time_ns() - t0) / 1e3  # μs
    end
    return median(times)
end

# Space groups to benchmark (General/P lattice only)
const GROUPS = [
    (221, "Pm-3m",  48),
    (47,  "Pmmm",    8),
    (123, "P4/mmm", 16),
    (10,  "P2/m",    4),
    (2,   "P-1",     2),
]

function run_benchmark(N_val::Int)
    N = (N_val, N_val, N_val)
    println("\n" * "="^80)
    @printf("  Grid N = %d³,  Subgrid M = %d³\n", N_val, N_val ÷ 2)
    println("="^80)

    @printf("%-10s  %6s  %9s  %9s  %7s  %9s  %9s  %7s\n",
            "Group", "|G|",
            "cfft μs", "rcfft μs", "fwd×",
            "icfft μs", "ircfft μs", "bwd×")
    println("-"^80)

    for (sg, name, order) in GROUPS
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        u = make_symmetric(ops_s, N)

        # ── Complex plans ────────────────────────────────
        pair_c = plan_cfft_pair(N, sg, 3; method=:general)
        L_c = stride_factors(pair_c)
        M_c = subgrid_size(pair_c)
        f0 = extract_subgrid(u, N, collect(L_c))
        n_spec = cfft_asu_size(pair_c)

        fwd_c = pair_c.fwd
        M_vol = prod(M_c)
        @inbounds for i in 1:M_vol
            fwd_c.input_buffer[i] = complex(f0[i])
        end

        F̂_c = Vector{ComplexF64}(undef, n_spec)
        CrystallographicFFT.fft_reconstruct!(fwd_c)
        @inbounds for i in 1:n_spec
            F̂_c[i] = fwd_c.output_buffer[i]
        end
        bwd_c = pair_c.bwd

        # ── Real plans ───────────────────────────────────
        pair_r = plan_rcfft_pair(N, sg, 3)
        fwd_r = pair_r.fwd
        M_vol_r = prod(fwd_r.M)
        @inbounds for i in 1:M_vol_r
            fwd_r.input_buffer[i] = f0[i]
        end

        F̂_r = Vector{ComplexF64}(undef, n_spec)
        CrystallographicFFT.rfft_reconstruct!(fwd_r)
        @inbounds for i in 1:n_spec
            F̂_r[i] = fwd_r.output_buffer[i]
        end
        bwd_r = pair_r.bwd

        # ── Benchmark forward ────────────────────────────
        t_fwd_c = bench(CrystallographicFFT.fft_reconstruct!, fwd_c)
        t_fwd_r = bench(CrystallographicFFT.rfft_reconstruct!, fwd_r)

        # ── Benchmark backward ───────────────────────────
        t_bwd_c = bench(CrystallographicFFT.execute_backward!, bwd_c, F̂_c)
        t_bwd_r = bench(CrystallographicFFT.execute_real_backward!, bwd_r, F̂_r)

        speedup_fwd = t_fwd_c / t_fwd_r
        speedup_bwd = t_bwd_c / t_bwd_r

        @printf("%-10s  %6d  %9.1f  %9.1f  %6.2f×  %9.1f  %9.1f  %6.2f×\n",
                name, order,
                t_fwd_c, t_fwd_r, speedup_fwd,
                t_bwd_c, t_bwd_r, speedup_bwd)
    end

    # Raw FFT baseline
    M_val = N_val ÷ 2
    M = (M_val, M_val, M_val)
    buf_c = zeros(ComplexF64, M...)
    buf_r = zeros(Float64, M...)
    p_fft = plan_fft(buf_c)
    p_rfft = plan_rfft(buf_r)
    out_c = similar(buf_c)
    M̂1 = M_val ÷ 2 + 1
    out_r = zeros(ComplexF64, M̂1, M_val, M_val)

    t_fft  = bench(mul!, out_c, p_fft, buf_c)
    t_rfft = bench(mul!, out_r, p_rfft, buf_r)

    println("-"^80)
    @printf("Raw FFT baseline (M=%d³):  fft = %.1f μs,  rfft = %.1f μs,  ratio = %.2f×\n",
            M_val, t_fft, t_rfft, t_fft / t_rfft)
end

for N_val in [32, 64, 128]
    run_benchmark(N_val)
end
