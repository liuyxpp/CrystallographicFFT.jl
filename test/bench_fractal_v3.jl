# Fractal KRFFT v3 Benchmark
# Compares v3 (ASU-only butterfly) against G0 ASU and Full FFT.
#
# IMPORTANT: Fractal v3 uses ORIGINAL ops (shift handled internally by shift_ops_half_grid).
#            G0 ASU uses find_optimal_shift externally.
#
# Usage: julia --project=test test/bench_fractal_v3.jl
#
using CrystallographicFFT
using CrystallographicFFT.SymmetryOps: get_ops
using CrystallographicFFT.ASU: find_optimal_shift
using CrystallographicFFT.SpectralIndexing: calc_spectral_asu
using CrystallographicFFT.KRFFT: plan_fractal_krfft, execute_fractal_krfft!,
    plan_fractal_krfft_v3, execute_fractal_krfft_v3!,
    plan_krfft_g0asu, execute_g0asu_krfft!
using FFTW
using Statistics
using LinearAlgebra: mul!

"""Benchmark a callable, return median time in ms."""
function bench(f; n_warmup=5, n_trials=30)
    for _ in 1:n_warmup; f(); end
    times = [(@elapsed f()) for _ in 1:n_trials]
    return median(times) * 1000  # ms
end

function benchmark_all(sg, name, N; n_warmup=5, n_trials=30)
    dim = length(N)
    ops_orig = get_ops(sg, dim, N)
    u = randn(Float64, N)

    # --- Full FFT baseline (in-place mul! for fair comparison) ---
    u_c = complex(u)
    fft_plan = plan_fft(u_c)
    F_full = similar(u_c)
    t_fft = bench(; n_warmup, n_trials) do
        mul!(F_full, fft_plan, u_c)
    end

    # --- G0 ASU: uses find_optimal_shift externally ---
    t_g0 = NaN
    try
        _, ops_shifted = find_optimal_shift(ops_orig, N)
        spec_shifted = calc_spectral_asu(ops_shifted, dim, N)
        plan_g0 = plan_krfft_g0asu(spec_shifted, ops_shifted)
        t_g0 = bench(; n_warmup, n_trials) do
            execute_g0asu_krfft!(plan_g0, spec_shifted, u)
        end
    catch e
        @warn "G0 ASU failed for $name: $e"
    end

    # --- v3: uses ORIGINAL ops (shift handled internally) ---
    spec_orig = calc_spectral_asu(ops_orig, dim, N)
    plan_v3 = plan_fractal_krfft_v3(spec_orig, ops_orig)
    t_v3 = bench(; n_warmup, n_trials) do
        execute_fractal_krfft_v3!(plan_v3, u)
    end

    # --- Correctness: v3 vs v1 (both use shifted ops internally) ---
    # Note: fractal KRFFT requires symmetric input (orbit equivalence).
    # For performance benchmarking, random input is fine (same work done).
    plan_v1 = plan_fractal_krfft(spec_orig, ops_orig)
    out_v1 = execute_fractal_krfft!(plan_v1, u)
    out_v3 = execute_fractal_krfft_v3!(plan_v3, u)
    max_err = maximum(abs.(out_v3 .- out_v1))

    # Print results
    function fmt(t, t_ref)
        isnan(t) && return "N/A"
        return "$(round(t, digits=3))ms ($(round(t_ref/t, digits=1))×)"
    end

    println("  $name |G|=$(length(ops_orig)): n_spec=$(length(spec_orig.points)) v3_vs_v1=$(round(max_err, sigdigits=2))")
    println("    FFT:    $(round(t_fft, digits=3))ms")
    println("    G0 ASU: $(fmt(t_g0, t_fft))")
    println("    v3:     $(fmt(t_v3, t_fft))")
    if !isnan(t_g0)
        ratio = t_v3 / t_g0
        println("    v3/G0:  $(round(ratio, digits=2))× $(ratio < 1 ? "✓ v3 faster" : "✗ G0 faster")")
    end
end

# --- Main ---
for N_size in [64, 128]
    N = (N_size, N_size, N_size)
    println("\n" * "="^60)
    println("N = $N_size")
    println("="^60)

    for (sg, name) in [(221, "Pm-3m"), (200, "Pm-3"), (225, "Fm-3m"), (229, "Im-3m")]
        try
            benchmark_all(sg, name, N)
        catch e
            @warn "Benchmark failed for $name N=$N_size" exception=(e, catch_backtrace())
        end
    end
end
