"""
GPU Benchmark: CrystallographicFFT vs Full-Grid CUFFT/FFTW

Correct comparison: CFFT operates on M-subgrid, baseline is FFT on FULL N-grid.
  CFFT speedup = FFT(N³) / CFFT_time
  Theoretical target: ≈ |G|× for forward, close to that for backward

Metrics:
  1. GPU CFFT speedup = CUFFT(N³) / GPU_cfft_time   → target ≈ |G|×
  2. CPU CFFT speedup = FFTW(N³) / CPU_cfft_time     → reference
  3. GPU vs CPU = CPU_cfft / GPU_cfft                 → GPU advantage
  4. GPU speedup / CPU speedup ratio                  → overhead parity

Run:  julia --project=test test/bench_gpu.jl
"""

using CrystallographicFFT
using CUDA
using FFTW
using LinearAlgebra
using Printf
using Statistics

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Benchmark GPU: warm up, synchronize, time with median."""
function bench_gpu(f; n_warmup=5, n_trials=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    times = Float64[]
    for _ in 1:n_trials
        CUDA.synchronize()
        t0 = time_ns()
        f()
        CUDA.synchronize()
        push!(times, (time_ns() - t0) * 1e-9)
    end
    return median(times)
end

"""Benchmark CPU: warm up, time with median."""
function bench_cpu(f; n_warmup=5, n_trials=20)
    for _ in 1:n_warmup; f(); end
    times = Float64[]
    for _ in 1:n_trials
        t0 = time_ns()
        f()
        push!(times, (time_ns() - t0) * 1e-9)
    end
    return median(times)
end

# ── Per-group benchmark ──────────────────────────────────────────────────────

function benchmark_group(sg, name, N_size, method;
                         n_warmup=5, n_trials=20)
    N = ntuple(_ -> N_size, 3)

    # 1. Create CFFT plans
    plan_gpu = plan_cfft_pair(N, sg, 3; method=method,
                               array_type=CuArray{Float64})
    plan_cpu = plan_cfft_pair(N, sg, 3; method=method)

    M = plan_cpu.M
    n_spec = plan_cpu.n_spec
    nG = length(plan_cpu.spec_asu.ops)  # |G|

    # 2. CFFT data (on M-grid)
    f0_cpu = randn(M...)
    F_cpu = zeros(ComplexF64, n_spec)
    f0_back_cpu = zeros(Float64, M...)
    f0_gpu = CuArray(f0_cpu)
    F_gpu = CUDA.zeros(ComplexF64, n_spec)
    f0_back_gpu = CUDA.zeros(Float64, M...)

    # === BASELINES: Full N-grid FFT ===

    # CPU: FFTW on FULL N-grid
    FFTW.set_num_threads(1)
    u_full_cpu = complex(randn(N...))
    u_full_out_cpu = similar(u_full_cpu)
    fftw_fwd_plan = plan_fft(u_full_cpu)
    ifftw_bwd_plan = plan_ifft(u_full_out_cpu)

    t_fftw_fwd = bench_cpu(() -> mul!(u_full_out_cpu, fftw_fwd_plan, u_full_cpu);
                            n_warmup=n_warmup, n_trials=n_trials)
    mul!(u_full_out_cpu, fftw_fwd_plan, u_full_cpu)
    t_fftw_bwd = bench_cpu(() -> mul!(u_full_cpu, ifftw_bwd_plan, u_full_out_cpu);
                            n_warmup=n_warmup, n_trials=n_trials)

    # GPU: CUFFT on FULL N-grid
    u_full_gpu = CuArray(u_full_cpu)
    u_full_out_gpu = similar(u_full_gpu)
    cufft_fwd_plan = plan_fft(u_full_gpu)
    cufft_bwd_plan = plan_ifft(u_full_out_gpu)

    t_cufft_fwd = bench_gpu(() -> mul!(u_full_out_gpu, cufft_fwd_plan, u_full_gpu);
                              n_warmup=n_warmup, n_trials=n_trials)
    mul!(u_full_out_gpu, cufft_fwd_plan, u_full_gpu)
    t_cufft_bwd = bench_gpu(() -> mul!(u_full_gpu, cufft_bwd_plan, u_full_out_gpu);
                              n_warmup=n_warmup, n_trials=n_trials)

    # 4. CFFT benchmarks
    t_gpu_fwd = bench_gpu(() -> cfft!(F_gpu, plan_gpu, f0_gpu);
                            n_warmup=n_warmup, n_trials=n_trials)
    cfft!(F_gpu, plan_gpu, f0_gpu)
    t_gpu_bwd = bench_gpu(() -> icfft!(f0_back_gpu, plan_gpu, F_gpu);
                            n_warmup=n_warmup, n_trials=n_trials)

    t_cpu_fwd = bench_cpu(() -> cfft!(F_cpu, plan_cpu, f0_cpu);
                            n_warmup=n_warmup, n_trials=n_trials)
    cfft!(F_cpu, plan_cpu, f0_cpu)
    t_cpu_bwd = bench_cpu(() -> icfft!(f0_back_cpu, plan_cpu, F_cpu);
                            n_warmup=n_warmup, n_trials=n_trials)

    # 5. Compute speedup ratios (vs FULL N-grid baselines)
    cpu_spd_fwd = t_fftw_fwd / t_cpu_fwd
    cpu_spd_bwd = t_fftw_bwd / t_cpu_bwd
    gpu_spd_fwd = t_cufft_fwd / t_gpu_fwd
    gpu_spd_bwd = t_cufft_bwd / t_gpu_bwd

    # GPU vs CPU absolute
    gpu_vs_cpu_fwd = t_cpu_fwd / t_gpu_fwd
    gpu_vs_cpu_bwd = t_cpu_bwd / t_gpu_bwd

    # GPU speedup / CPU speedup ratio
    parity_fwd = gpu_spd_fwd / cpu_spd_fwd
    parity_bwd = gpu_spd_bwd / cpu_spd_bwd

    return (
        name=name, sg=sg, method=method, N=N_size,
        M=M, n_spec=n_spec, nG=nG,
        # Absolute times
        t_fftw_fwd=t_fftw_fwd, t_fftw_bwd=t_fftw_bwd,
        t_cufft_fwd=t_cufft_fwd, t_cufft_bwd=t_cufft_bwd,
        t_cpu_fwd=t_cpu_fwd, t_cpu_bwd=t_cpu_bwd,
        t_gpu_fwd=t_gpu_fwd, t_gpu_bwd=t_gpu_bwd,
        # Speedups vs full-grid baseline
        cpu_spd_fwd=cpu_spd_fwd, cpu_spd_bwd=cpu_spd_bwd,
        gpu_spd_fwd=gpu_spd_fwd, gpu_spd_bwd=gpu_spd_bwd,
        # GPU vs CPU
        gpu_vs_cpu_fwd=gpu_vs_cpu_fwd, gpu_vs_cpu_bwd=gpu_vs_cpu_bwd,
        # Parity
        parity_fwd=parity_fwd, parity_bwd=parity_bwd,
    )
end

function main()
    @info "GPU: $(CUDA.name(CUDA.device()))"
    @info "FFTW threads: 1 (single-threaded baseline)"

    println()
    println("=" ^ 140)
    println("GPU Benchmark: CFFT Speedup vs Full-Grid FFT(N³)")
    println("  Speedup = FFT(N³) / CFFT_time  |  Target ≈ |G|×")
    println("=" ^ 140)

    test_cases = [
        (225, "Pm-3m",  :general),
        (200, "Pm-3",   :general),
        (225, "Fm-3m",  :centered),
        (229, "Im-3m",  :centered),
    ]

    all_results = []

    for N_size in [32, 64, 128]
        n_trials = N_size <= 64 ? 30 : 10
        n_warmup = 5

        println()
        println("─" ^ 140)
        @printf("N = %d  (warmup=%d, trials=%d)\n", N_size, n_warmup, n_trials)
        println("─" ^ 140)

        # Table 1: Absolute times
        println()
        @printf("%-8s %-4s %-9s %-10s %4s │ %8s %8s %8s %8s │ %8s %8s %8s %8s\n",
                "Group", "SG", "Method", "M", "|G|",
                "FFTW_f", "FFTW_b", "CPU_f", "CPU_b",
                "CUFFT_f", "CUFFT_b", "GPU_f", "GPU_b")
        println("-" ^ 120)

        results_n = []
        for (sg, name, method) in test_cases
            r = benchmark_group(sg, name, N_size, method;
                                n_warmup=n_warmup, n_trials=n_trials)
            push!(results_n, r)
            push!(all_results, r)

            M_str = join(r.M, "×")
            @printf("%-8s %-4d %-9s %-10s %4d │ %7.3f  %7.3f  %7.3f  %7.3f │ %7.3f  %7.3f  %7.3f  %7.3f\n",
                    name, sg, method, M_str, r.nG,
                    r.t_fftw_fwd*1e3, r.t_fftw_bwd*1e3,
                    r.t_cpu_fwd*1e3, r.t_cpu_bwd*1e3,
                    r.t_cufft_fwd*1e3, r.t_cufft_bwd*1e3,
                    r.t_gpu_fwd*1e3, r.t_gpu_bwd*1e3)
        end

        # Table 2: Speedups vs full-grid FFT(N³)
        println()
        @printf("%-8s %-4s %-9s %4s │ CPU_fwd   CPU_bwd   │ GPU_fwd   GPU_bwd   │ GPU/CPU_f GPU/CPU_b │ parity_f parity_b\n",
                "Group", "SG", "Method", "|G|")
        println("-" ^ 130)
        for r in results_n
            @printf("%-8s %-4d %-9s %4d │ %6.2f×   %6.2f×   │ %6.2f×   %6.2f×   │ %7.1f×   %7.1f×  │ %6.2f    %6.2f\n",
                    r.name, r.sg, r.method, r.nG,
                    r.cpu_spd_fwd, r.cpu_spd_bwd,
                    r.gpu_spd_fwd, r.gpu_spd_bwd,
                    r.gpu_vs_cpu_fwd, r.gpu_vs_cpu_bwd,
                    r.parity_fwd, r.parity_bwd)
        end
    end

    # Cross-size summary
    println()
    println("=" ^ 140)
    println("Cross-Size: CFFT Speedup vs Full-Grid FFT(N³)")
    println("=" ^ 140)
    @printf("%-8s %-9s %4s │", "Group", "Method", "|G|")
    for N_size in [32, 64, 128]
        @printf("  CPU_f(N=%-3d) GPU_f(N=%-3d)  │", N_size, N_size)
    end
    println()
    println("-" ^ 120)
    for (sg, name, method) in test_cases
        rs = filter(r -> r.sg == sg && r.method == method, all_results)
        nG = isempty(rs) ? 0 : rs[1].nG
        @printf("%-8s %-9s %4d │", name, method, nG)
        for N_size in [32, 64, 128]
            rf = filter(r -> r.N == N_size, rs)
            if !isempty(rf)
                r = rf[1]
                @printf("  %9.2f×   %9.2f×   │",
                        r.cpu_spd_fwd, r.gpu_spd_fwd)
            else
                @printf("  %9s   %9s   │", "—", "—")
            end
        end
        println()
    end
    println("=" ^ 140)
end

main()
