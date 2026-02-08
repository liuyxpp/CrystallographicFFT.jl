# KRFFT Reconstruction Method Benchmark
# Compares 5 methods: Full FFT, Butterfly, Sparse, Selective G0, G0 ASU
#
# Usage: julia --project=test test/bench_krfft_methods.jl
#
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_krfft_recursive, execute_recursive_krfft!,
    plan_krfft_sparse, execute_sparse_krfft!,
    plan_krfft_selective, execute_selective_krfft!,
    plan_krfft_g0asu, execute_g0asu_krfft!
using FFTW
using Statistics
using LinearAlgebra: mul!

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

"""Benchmark a single callable, return median time in ms."""
function bench_method(f; n_warmup=5, n_trials=30)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    return median(times) * 1000  # ms
end

function benchmark_group(sg, name, N; n_warmup=5, n_trials=30)
    dim = length(N)
    ops = get_ops(sg, dim, N)
    _, ops_s = find_optimal_shift(ops, N)
    spec = calc_spectral_asu(ops_s, dim, N)
    n_spec = length(spec.points)
    u = make_symmetric_field(ops_s, N)

    # --- Full FFT baseline ---
    u_c = complex(u)
    fft_plan = plan_fft(u_c)
    F_full = similar(u_c)
    t_fft = bench_method(; n_warmup, n_trials) do
        mul!(F_full, fft_plan, u_c)
    end

    # --- Butterfly ---
    p_but = plan_krfft_recursive(spec, ops_s)
    t_but = bench_method(; n_warmup, n_trials) do
        execute_recursive_krfft!(p_but, spec, u)
    end

    # --- Sparse ---
    p_sp = plan_krfft_sparse(spec, ops_s)
    t_sp = bench_method(; n_warmup, n_trials) do
        execute_sparse_krfft!(p_sp, spec, u)
    end

    # --- Selective G0 ---
    p_sel = plan_krfft_selective(spec, ops_s)
    t_sel = bench_method(; n_warmup, n_trials) do
        execute_selective_krfft!(p_sel, spec, u)
    end

    # --- G0 ASU ---
    p_asu = plan_krfft_g0asu(spec, ops_s)
    t_asu = bench_method(; n_warmup, n_trials) do
        execute_g0asu_krfft!(p_asu, spec, u)
    end

    GC.gc()

    return (sg=sg, name=name, G=length(ops_s), n_spec=n_spec, n_reps=p_asu.n_reps,
            t_fft=t_fft, t_but=t_but, t_sp=t_sp, t_sel=t_sel, t_asu=t_asu)
end

function run_benchmarks()
    test_groups = [
        (225, "Fm-3m"),   # |G|=192, cubic
        (227, "Fd-3m"),   # |G|=192, cubic (diamond)
        (229, "Im-3m"),   # |G|=96, cubic
        (221, "Pm-3m"),   # |G|=48, cubic
        (200, "Pm-3"),    # |G|=24, cubic
        (136, "P42/mnm"), # |G|=16, tetragonal
        (70,  "Fddd"),    # |G|=16, orthorhombic
    ]

    for N_size in [64]
        N = (N_size, N_size, N_size)
        M = N_size ÷ 2

        println("\n" * "="^110)
        println("Grid N = $N, M = $M, M³ = $(M^3), M2 = $(M÷2), 4×M2³ = $(4*(M÷2)^3)")
        println("Warmup: 5, Trials: 30, Metric: median")
        println("="^110)

        # Header
        println()
        header = rpad("SG", 6) * rpad("Name", 8) * rpad("|G|", 5) *
                 rpad("n_spec", 7) * rpad("n_reps", 7) *
                 rpad("FFT(ms)", 10) *
                 rpad("Bfly(ms)", 10) * rpad("Bfly_x", 8) *
                 rpad("Spar(ms)", 10) * rpad("Spar_x", 8) *
                 rpad("Sel(ms)", 10) * rpad("Sel_x", 8) *
                 rpad("ASU(ms)", 10) * rpad("ASU_x", 8)
        println(header)
        println("-"^110)

        results = []
        for (sg, name) in test_groups
            try
                r = benchmark_group(sg, name, N)
                push!(results, r)

                line = rpad(string(r.sg), 6) *
                       rpad(r.name, 8) *
                       rpad(string(r.G), 5) *
                       rpad(string(r.n_spec), 7) *
                       rpad(string(r.n_reps), 7) *
                       rpad(string(round(r.t_fft, digits=2)), 10) *
                       rpad(string(round(r.t_but, digits=3)), 10) *
                       rpad(string(round(r.t_fft/r.t_but, digits=1)) * "x", 8) *
                       rpad(string(round(r.t_sp, digits=3)), 10) *
                       rpad(string(round(r.t_fft/r.t_sp, digits=1)) * "x", 8) *
                       rpad(string(round(r.t_sel, digits=3)), 10) *
                       rpad(string(round(r.t_fft/r.t_sel, digits=1)) * "x", 8) *
                       rpad(string(round(r.t_asu, digits=3)), 10) *
                       rpad(string(round(r.t_fft/r.t_asu, digits=1)) * "x", 8)
                println(line)
            catch e
                println(rpad(string(sg), 6) * rpad(name, 8) * "ERROR: $e")
            end
        end

        # Summary: best method per group
        println("\n--- Best method per group ---")
        for r in results
            methods = [:t_but => "Butterfly", :t_sp => "Sparse",
                       :t_sel => "Selective", :t_asu => "G0 ASU"]
            best_name = ""
            best_t = Inf
            for (sym, nm) in methods
                t = getfield(r, sym)
                if t < best_t
                    best_t = t
                    best_name = nm
                end
            end
            speedup = r.t_fft / best_t
            println("  $(rpad(r.name, 8)): $(best_name) @ $(round(best_t, digits=3)) ms ($(round(speedup, digits=1))x)")
        end

        # Markdown table for documentation
        println("\n--- Markdown table ---")
        println("| Group | |G| | n_spec | n_reps | Full FFT | Butterfly | Sparse | Selective | **G0 ASU** |")
        println("|-------|-----|--------|--------|----------|-----------|--------|-----------|------------|")
        for r in results
            s(t) = "$(round(t, digits=3)) ms ($(round(r.t_fft/t, digits=1))x)"
            println("| $(r.name) | $(r.G) | $(r.n_spec) | $(r.n_reps) | $(round(r.t_fft, digits=2)) ms | $(s(r.t_but)) | $(s(r.t_sp)) | $(s(r.t_sel)) | **$(s(r.t_asu))** |")
        end
    end
end

run_benchmarks()
