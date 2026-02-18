"""
GPU Test & Benchmark: rcfft!/ircfft!

Verifies:
  1. GPU plan creation (plan_rcfft_pair with CuArray)
  2. Roundtrip correctness: rcfft! → ircfft! ≈ identity on GPU
  3. GPU ≈ CPU cross-validation: GPU and CPU rcfft! produce identical spectra
  4. Performance: GPU rcfft!/ircfft! vs GPU cfft!/icfft! vs CPU rcfft!/ircfft!

Run:  julia --project=test test/test_rcfft_gpu.jl
"""

using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.CFFTApi
using CUDA
using FFTW
using LinearAlgebra
using Printf
using Statistics

include("test_helpers.jl")

# ── Helpers ──────────────────────────────────────────────────────────────────

function bench_gpu(f; n_warmup=5, n_trials=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    times = Float64[]
    for _ in 1:n_trials
        CUDA.synchronize()
        t0 = time_ns()
        f()
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e3)  # μs
    end
    return median(times)
end

function bench_cpu(f; n_warmup=5, n_trials=20)
    for _ in 1:n_warmup; f(); end
    times = Float64[]
    for _ in 1:n_trials
        t0 = time_ns()
        f()
        push!(times, (time_ns() - t0) / 1e3)  # μs
    end
    return median(times)
end

# ── Space groups ─────────────────────────────────────────────────────────────

const GROUPS = [
    (221, "Pm-3m",  48),
    (47,  "Pmmm",    8),
    (123, "P4/mmm", 16),
    (2,   "P-1",     2),
]

# ── Correctness Tests ────────────────────────────────────────────────────────

function test_correctness(N_val::Int)
    N = (N_val, N_val, N_val)
    println("\n" * "="^80)
    @printf("  Correctness Tests: N = %d³\n", N_val)
    println("="^80)

    n_pass = 0
    n_fail = 0

    for (sg, name, order) in GROUPS
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        u = make_symmetric(ops_s, N)

        # CPU plan
        pair_cpu = plan_rcfft_pair(N, sg, 3)
        L = stride_factors(pair_cpu)
        M = subgrid_size(pair_cpu)
        f0_cpu = extract_subgrid(u, N, collect(L))
        n_spec = cfft_asu_size(pair_cpu)

        # GPU plan
        pair_gpu = plan_rcfft_pair(N, sg, 3; array_type=CuArray{Float64})

        # GPU data
        f0_gpu = CuArray(f0_cpu)
        F̂_gpu = CUDA.zeros(ComplexF64, n_spec)
        f0_out_gpu = CUDA.zeros(Float64, M...)

        # Test 1: Roundtrip on GPU
        rcfft!(F̂_gpu, pair_gpu, f0_gpu)
        ircfft!(f0_out_gpu, pair_gpu, F̂_gpu)

        f0_out_host = Array(f0_out_gpu)
        err_rt = maximum(abs.(f0_cpu .- f0_out_host))

        # Test 2: GPU ≈ CPU forward
        F̂_cpu = Vector{ComplexF64}(undef, n_spec)
        rcfft!(F̂_cpu, pair_cpu, f0_cpu)

        F̂_gpu_host = Array(F̂_gpu)
        err_fwd = maximum(abs.(F̂_cpu .- F̂_gpu_host))

        # Test 3: GPU backward ≈ CPU backward
        f0_bwd_cpu = zeros(M...)
        ircfft!(f0_bwd_cpu, pair_cpu, F̂_cpu)

        err_bwd = maximum(abs.(f0_bwd_cpu .- f0_out_host))

        rt_ok  = err_rt  < 1e-10 ? "✓" : "✗"
        fwd_ok = err_fwd < 1e-10 ? "✓" : "✗"
        bwd_ok = err_bwd < 1e-10 ? "✓" : "✗"

        if err_rt < 1e-10 && err_fwd < 1e-10 && err_bwd < 1e-10
            n_pass += 3
        else
            n_fail += (err_rt >= 1e-10 ? 1 : 0) + (err_fwd >= 1e-10 ? 1 : 0) + (err_bwd >= 1e-10 ? 1 : 0)
            n_pass += (err_rt < 1e-10 ? 1 : 0) + (err_fwd < 1e-10 ? 1 : 0) + (err_bwd < 1e-10 ? 1 : 0)
        end

        @printf("  %-10s (SG%3d):  roundtrip=%s (%.1e)  fwd=%s (%.1e)  bwd=%s (%.1e)\n",
                name, sg, rt_ok, err_rt, fwd_ok, err_fwd, bwd_ok, err_bwd)
    end

    println("-"^80)
    @printf("  Results: %d passed, %d failed\n", n_pass, n_fail)
    return n_fail
end

# ── Performance Benchmark ────────────────────────────────────────────────────

function benchmark_performance(N_val::Int)
    N = (N_val, N_val, N_val)
    println("\n" * "="^80)
    @printf("  Performance Benchmark: N = %d³\n", N_val)
    println("="^80)

    n_trials = N_val <= 64 ? 30 : 10

    @printf("%-10s  %4s  %9s  %9s  %7s  %9s  %9s  %7s  %8s  %8s\n",
            "Group", "|G|",
            "GPU_rc μs", "GPU_c μs", "rc/c×",
            "GPU_irc μs", "GPU_ic μs", "irc/ic×",
            "CPU_rc μs", "GPU/CPU×")
    println("-"^110)

    for (sg, name, order) in GROUPS
        ops = get_ops(sg, 3, N)
        _, ops_s = find_optimal_shift(ops, N)
        u = make_symmetric(ops_s, N)

        # Plans
        pair_r_gpu = plan_rcfft_pair(N, sg, 3; array_type=CuArray{Float64})
        pair_c_gpu = plan_cfft_pair(N, sg, 3; method=:general, array_type=CuArray{Float64})
        pair_r_cpu = plan_rcfft_pair(N, sg, 3)

        L = stride_factors(pair_r_cpu)
        M = subgrid_size(pair_r_cpu)
        f0_cpu = extract_subgrid(u, N, collect(L))
        n_spec = cfft_asu_size(pair_r_cpu)

        # GPU buffers
        f0_gpu = CuArray(f0_cpu)
        F̂_r_gpu = CUDA.zeros(ComplexF64, n_spec)
        F̂_c_gpu = CUDA.zeros(ComplexF64, n_spec)
        f0_out_r_gpu = CUDA.zeros(Float64, M...)
        f0_out_c_gpu = CUDA.zeros(Float64, M...)

        # CPU buffers
        F̂_r_cpu = Vector{ComplexF64}(undef, n_spec)
        f0_out_r_cpu = zeros(M...)

        # Warmup
        rcfft!(F̂_r_gpu, pair_r_gpu, f0_gpu)
        cfft!(F̂_c_gpu, pair_c_gpu, f0_gpu)
        rcfft!(F̂_r_cpu, pair_r_cpu, f0_cpu)

        # Benchmark GPU rcfft
        t_gpu_rfwd = bench_gpu(() -> rcfft!(F̂_r_gpu, pair_r_gpu, f0_gpu);
                                n_trials=n_trials)
        rcfft!(F̂_r_gpu, pair_r_gpu, f0_gpu)
        t_gpu_rbwd = bench_gpu(() -> ircfft!(f0_out_r_gpu, pair_r_gpu, F̂_r_gpu);
                                n_trials=n_trials)

        # Benchmark GPU cfft (general)
        t_gpu_cfwd = bench_gpu(() -> cfft!(F̂_c_gpu, pair_c_gpu, f0_gpu);
                                n_trials=n_trials)
        cfft!(F̂_c_gpu, pair_c_gpu, f0_gpu)
        t_gpu_cbwd = bench_gpu(() -> icfft!(f0_out_c_gpu, pair_c_gpu, F̂_c_gpu);
                                n_trials=n_trials)

        # Benchmark CPU rcfft
        t_cpu_rfwd = bench_cpu(() -> rcfft!(F̂_r_cpu, pair_r_cpu, f0_cpu);
                                n_trials=n_trials)

        speedup_fwd = t_gpu_cfwd / t_gpu_rfwd
        speedup_bwd = t_gpu_cbwd / t_gpu_rbwd
        gpu_vs_cpu  = t_cpu_rfwd / t_gpu_rfwd

        @printf("%-10s  %4d  %9.1f  %9.1f  %6.2f×  %9.1f  %9.1f  %6.2f×  %8.1f  %7.1f×\n",
                name, order,
                t_gpu_rfwd, t_gpu_cfwd, speedup_fwd,
                t_gpu_rbwd, t_gpu_cbwd, speedup_bwd,
                t_cpu_rfwd, gpu_vs_cpu)
    end
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    @info "GPU: $(CUDA.name(CUDA.device()))"
    FFTW.set_num_threads(1)

    # Correctness
    total_fail = 0
    for N_val in [16, 32]
        total_fail += test_correctness(N_val)
    end

    if total_fail > 0
        @error "GPU correctness tests FAILED ($total_fail failures)"
        return
    end
    @info "All GPU correctness tests passed ✓"

    # Performance
    for N_val in [32, 64, 128]
        benchmark_performance(N_val)
    end
end

main()
