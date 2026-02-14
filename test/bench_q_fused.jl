# M2 Q-Fused KRFFT Benchmark
#
# Benchmarks the Q-fused hot path (FFT → Q·Y → IFFT on M-grid)
# against full-grid FFT→K→IFFT baseline.
#
# Key: only the hot path is timed — no grid conversion overhead.
#
# Usage: julia --project=test test/bench_q_fused.jl
#
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.QFusedKRFFT
using FFTW
using Statistics
using LinearAlgebra

"""Build a symmetrized real-space field for the given (shifted) symmetry operations."""
function make_symmetric_field(ops, N)
    u = rand(N...)
    u_sym = zeros(N...)
    R_mats = [Int.(op.R) for op in ops]
    t_vecs = [round.(Int, op.t) for op in ops]
    Nv = collect(Int, N)
    @inbounds for k in 1:N[3], j in 1:N[2], i in 1:N[1]
        x1, x2, x3 = i-1, j-1, k-1
        for g in eachindex(ops)
            R = R_mats[g]; t = t_vecs[g]
            y1 = mod(R[1,1]*x1 + R[1,2]*x2 + R[1,3]*x3 + t[1], Nv[1]) + 1
            y2 = mod(R[2,1]*x1 + R[2,2]*x2 + R[2,3]*x3 + t[2], Nv[2]) + 1
            y3 = mod(R[3,1]*x1 + R[3,2]*x2 + R[3,3]*x3 + t[3], Nv[3]) + 1
            u_sym[i,j,k] += u[y1, y2, y3]
        end
    end
    u_sym ./= length(ops)
    return u_sym
end

"""Benchmark a callable, return median time in ms."""
function bench_method(f; n_warmup=5, n_trials=30)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    return median(times) * 1000  # ms
end

function benchmark_q_fused(sg, name, crystal_system, N, Δs, lattice;
                           n_warmup=5, n_trials=30)
    dim = 3
    ops = get_ops(sg, dim, N)
    _, ops_s = find_optimal_shift(ops, N)
    u_sym = make_symmetric_field(ops_s, N)

    # --- Full FFT → K → IFFT baseline ---
    recip_B = 2π * inv(lattice)'
    u_c = complex(u_sym)
    fft_p = plan_fft!(u_c)
    ifft_p = plan_ifft!(u_c)
    K_grid = Array{Float64}(undef, N)
    for ci in CartesianIndices(K_grid)
        h = [ci[d]-1 for d in 1:3]
        for d in 1:3; h[d] >= N[d]÷2 && (h[d] -= N[d]); end
        Kvec = recip_B * h
        K_grid[ci] = exp(-dot(Kvec, Kvec) * Δs)
    end
    F_buf = similar(u_c)

    t_baseline = bench_method(; n_warmup, n_trials) do
        copyto!(F_buf, complex.(u_sym))
        fft_p * F_buf
        @. F_buf *= K_grid
        ifft_p * F_buf
    end

    # --- FFT-only baseline (no K multiply, no IFFT) ---
    t_fft_only = bench_method(; n_warmup, n_trials) do
        copyto!(F_buf, complex.(u_sym))
        fft_p * F_buf
    end

    # --- Q-fused hot path ---
    plan = plan_m2_q(N, sg, dim, Δs, lattice)
    L = plan.L
    M = plan.M
    d = prod(L)

    f0 = zeros(Float64, Tuple(M))
    fullgrid_to_subgrid!(f0, u_sym, plan)
    f0_work = copy(f0)

    t_qfused = bench_method(; n_warmup, n_trials) do
        copyto!(f0_work, f0)
        execute_m2_q!(plan, f0_work)
    end

    # --- Breakdown: subgrid FFT-only time ---
    Y_tmp = zeros(ComplexF64, Tuple(M))
    sub_fft_p = plan_fft!(Y_tmp)

    t_subfft = bench_method(; n_warmup, n_trials) do
        @. Y_tmp = complex(f0)
        sub_fft_p * Y_tmp
    end

    GC.gc()

    speedup_vs_baseline = t_baseline / t_qfused
    speedup_vs_fft = t_fft_only / t_qfused

    return (sg=sg, name=name, system=crystal_system,
            G=length(ops_s), L=L, M=M, d=d,
            N_total=prod(N), M_total=prod(M),
            t_baseline=t_baseline, t_fft_only=t_fft_only,
            t_qfused=t_qfused, t_subfft=t_subfft,
            speedup_vs_baseline=speedup_vs_baseline,
            speedup_vs_fft=speedup_vs_fft)
end

function run_benchmarks()
    Δs = 0.05
    lattice = Matrix{Float64}(I, 3, 3)

    test_groups = [
        # Orthorhombic
        (47,  "Pmmm",    "orthorhombic"),
        (25,  "Pmm2",    "orthorhombic"),
        (16,  "P222",    "orthorhombic"),
        (70,  "Fddd",    "orthorhombic"),

        # Tetragonal
        (123, "P4/mmm",  "tetragonal"),
        (136, "P42/mnm", "tetragonal"),
        (139, "I4/mmm",  "tetragonal"),

        # Cubic
        (200, "Pm-3",    "cubic"),
        (221, "Pm-3m",   "cubic"),
        (225, "Fm-3m",   "cubic"),
        (227, "Fd-3m",   "cubic"),
        (229, "Im-3m",   "cubic"),
        (230, "Ia-3d",   "cubic"),

        # Low symmetry
        (2,   "P-1",     "triclinic"),
        (10,  "P2/m",    "monoclinic"),
    ]

    for N_size in [64, 128]
        N = (N_size, N_size, N_size)

        println("\n" * "="^130)
        println("M2 Q-Fused KRFFT Benchmark: N = $N")
        println("FFTW threads: $(FFTW.get_num_threads()), Δs=$Δs, Warmup: 5, Trials: 30, Metric: median")
        println("="^130)

        header = rpad("SG", 5) * rpad("Name", 10) * rpad("System", 14) *
                 rpad("|G|", 5) * rpad("L", 12) * rpad("d", 4) *
                 rpad("M³", 12) *
                 rpad("FFT+K(ms)", 11) * rpad("FFT(ms)", 9) *
                 rpad("Q-fused(ms)", 13) * rpad("subFFT(ms)", 12) *
                 rpad("vs BL", 8) * rpad("vs FFT", 8)
        println("\n" * header)
        println("-"^130)

        results = []
        for (sg, name, system) in test_groups
            try
                r = benchmark_q_fused(sg, name, system, N, Δs, lattice)
                push!(results, r)

                line = rpad(string(r.sg), 5) *
                       rpad(r.name, 10) *
                       rpad(r.system, 14) *
                       rpad(string(r.G), 5) *
                       rpad(string(r.L), 12) *
                       rpad(string(r.d), 4) *
                       rpad(string(r.M_total), 12) *
                       rpad(string(round(r.t_baseline, digits=2)), 11) *
                       rpad(string(round(r.t_fft_only, digits=2)), 9) *
                       rpad(string(round(r.t_qfused, digits=3)), 13) *
                       rpad(string(round(r.t_subfft, digits=3)), 12) *
                       rpad(string(round(r.speedup_vs_baseline, digits=2)) * "x", 8) *
                       rpad(string(round(r.speedup_vs_fft, digits=2)) * "x", 8)
                println(line)
            catch e
                println(rpad(string(sg), 5) * rpad(name, 10) * rpad(system, 14) *
                        "ERROR: $(sprint(showerror, e))")
            end
        end

        # --- Summary ---
        println("\n--- Summary ---")
        if !isempty(results)
            best = argmax(r -> r.speedup_vs_baseline, results)
            worst = argmin(r -> r.speedup_vs_baseline, results)
            println("  Best:  $(best.name) (SG $(best.sg)) → $(round(best.speedup_vs_baseline, digits=2))x vs baseline")
            println("  Worst: $(worst.name) (SG $(worst.sg)) → $(round(worst.speedup_vs_baseline, digits=2))x vs baseline")
            avg_speedup = mean(r.speedup_vs_baseline for r in results)
            println("  Average speedup vs baseline: $(round(avg_speedup, digits=2))x")
        end

        # --- Markdown table ---
        println("\n--- Markdown table ---")
        println("| SG | Name | |G| | L | d | M³ | FFT+K (ms) | Q-fused (ms) | Speedup vs BL | Speedup vs FFT |")
        println("|-----|------|-----|---|---|-----|------------|--------------|---------------|----------------|")
        for r in results
            println("| $(r.sg) | $(r.name) | $(r.G) | $(r.L) | $(r.d) | $(r.M_total) " *
                    "| $(round(r.t_baseline, digits=2)) | $(round(r.t_qfused, digits=3)) " *
                    "| **$(round(r.speedup_vs_baseline, digits=2))x** " *
                    "| $(round(r.speedup_vs_fft, digits=2))x |")
        end
    end
end

FFTW.set_num_threads(1)
run_benchmarks()
